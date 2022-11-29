import transformers as tr
import torch

class UncertainityEstimator:

    def __call__(self, generation_output: tr.utils.ModelOutput) -> torch.Tensor:
        raise NotImplementedError
    

class BeamScoreMarginUncertaintyEstimator(UncertainityEstimator):

    def __init__(self, num_return_sequences: int, do_softmax: bool = True):
        self.num_return_sequences = num_return_sequences
        self.do_softmax = do_softmax

    def __call__(self, generation_output: tr.utils.ModelOutput) -> torch.Tensor:
        assert hasattr(generation_output, "sequences_scores"), "generation_output must have sequences_scores attribute"

        scores = generation_output.sequences_scores.view(-1, self.num_return_sequences)
        if self.do_softmax:
            scores = torch.softmax(scores, dim=1)
            margins = 1 - torch.abs(scores[:, 0] - scores[:, 1])
        else:
            margins = scores[:, 0] - scores[:, 1]
        return margins

