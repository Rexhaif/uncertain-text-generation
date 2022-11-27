import argparse as ap
from configparser import MAX_INTERPOLATION_DEPTH
import logging

import datasets as ds
import evaluate as ev
import torch
import transformers as tr
from rich.logging import RichHandler
from tqdm.auto import tqdm

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

    args = parser.parse_args()
    logger.info(f"Running evaluation with args: {args}")

    if args.use_8bit and args.use_16bit:
        raise ValueError("Cannot use both 8-bit and 16-bit quantization")

    logger.info("Loading dataset")
    dataset = ds.load_dataset(
        args.dataset_name, split=args.dataset_split
    )
    exact_match = ev.load("exact_match")

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

    collate_fn = tr.DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest", max_length=512, pad_to_multiple_of=8, return_tensors="pt"
    )
    data_loader = torch.utils.data.DataLoader(
        questions_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    model = model.eval()

    predictions = []
    for batch in tqdm(data_loader, unit='batch'):
        inputs = {
            k: v.to(model.device) for k, v in batch.items()
        }
        answer_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=args.max_new_tokens,
            num_beams=20,
            early_stopping=True,
            num_return_sequences=1,
            do_sample=False,
        )

        predictions += tokenizer.batch_decode(answer_ids, skip_special_tokens=True)

    logger.info("Computing metrics")
    value = exact_match.compute(predictions=predictions, references=answers_dataset)

    logger.info(f"Exact match: {value['exact_match']:.4f}")


if __name__ == "__main__":
    main()