import argparse as ap
import logging
import pandas as pd

import datasets as ds
import evaluate as ev
import torch
import transformers as tr
from rich.logging import RichHandler
from tqdm.auto import tqdm
import ue_methods as ue

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def main():
    parser = ap.ArgumentParser(
        prog="evaluate_seq2seq_qa.py",
        description="Evaluate a seq2seq model on a QA dataset",
    )
    parser.add_argument(
        "--model-name", type=str, required=True, help="Name of the model to evaluate"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset to evaluate on",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        required=True,
        help="Name of the dataset split to evaluate on",
    )
    parser.add_argument(
        "--question-column",
        type=str,
        required=True,
        help="Name of the column containing the questions",
    )
    parser.add_argument(
        "--answer-column",
        type=str,
        required=True,
        help="Name of the column containing the answers",
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        default=False,
        help="Use 8-bit quantization during evaluation",
    )
    parser.add_argument(
        "--use-16bit",
        action="store_true",
        default=False,
        help="Use 16-bit quantization during evaluation",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size to use during evaluation"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate per batch",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to add to the beginning of each question",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the file where to save the results",
    )

    args = parser.parse_args()
    logger.info(f"Running evaluation with args: {args}")

    if args.use_8bit and args.use_16bit:
        raise ValueError("Cannot use both 8-bit and 16-bit quantization")

    logger.info("Loading dataset")
    dataset = ds.load_dataset(
        args.dataset_name, split=args.dataset_split
    )

    logger.info("Loading model")
    tokenizer = tr.AutoTokenizer.from_pretrained(args.model_name)
    if args.use_8bit:
        model = tr.AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name, device_map='auto', load_in_8bit=True
        )
    elif args.use_16bit:
        model = tr.AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name, device_map='auto', torch_dtype=torch.float16
        )
    else:
        model = tr.AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name,
                device_map='auto'
        )

    logger.info("Preparing dataset")
    def prepare_function(examples):
        if args.prefix:
            questions = [args.prefix + q for q in examples[args.question_column]]
        else:
            questions = examples[args.question_column]

        return tokenizer(
            questions,
            truncation=True,
            max_length=512,
        )

    questions_dataset = dataset.map(
        prepare_function,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=dataset.column_names,
    )
    answers_dataset = dataset[args.answer_column]

    logger.info("Evaluating model")

    ue_estimator = ue.BeamScoreMarginUncertaintyEstimator(
        num_return_sequences=2,
        do_softmax=False,
    )

    collate_fn = tr.DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest", max_length=512, pad_to_multiple_of=8, return_tensors="pt"
    )
    data_loader = torch.utils.data.DataLoader(
        questions_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False
    )

    model = model.eval()

    results = []
    for batch in tqdm(data_loader, unit='batch'):
        inputs = {
            k: v.to(model.device) for k, v in batch.items()
        }
        generation_outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=args.max_new_tokens,
            num_beams=20,
            num_beam_groups=20,
            diversity_penalty=0.5,
            early_stopping=True,
            num_return_sequences=2,
            output_scores=True,
            return_dict_in_generate=True
        )
        ue_estimates = ue_estimator(generation_outputs)

        top_sequence = generation_outputs.sequences.view(inputs['input_ids'].shape[0], 2, -1)[:, 0, :]

        top_sequence = tokenizer.batch_decode(top_sequence, skip_special_tokens=True)

        results += [
            {
                "predicted_answer": top_sequence[i],
                "uncertainty_estimate": ue_estimates[i].item(),
            }
            for i in range(len(top_sequence))
        ]

    results = pd.DataFrame(results)
    results["true_answer"] = answers_dataset
    results["question"] = dataset[args.question_column]

    logger.info("Saving results")
    results.to_csv(args.output_file, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()