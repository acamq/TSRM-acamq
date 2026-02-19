import torch

from architecture.metrics import calc_metrics
from architecture.model import TSRMImputation


class TSRMImputationExternal(TSRMImputation):

    def _run(
        self,
        masked_data,
        original_data,
        embedding_x,
        embedding_y,
        determine_metrics=True,
        calc_real=False,
        phase="train",
    ):
        mask = torch.isnan(masked_data)
        input_data = torch.nan_to_num(masked_data, nan=0.0)

        output, _ = self.forward(input_data, embedding_x, mask=mask)

        loss = self.loss(prediction=output, target=original_data, mask=mask)
        if determine_metrics:
            metrics = calc_metrics(output=output, target=original_data, mask=mask)
            metrics.update({"loss": float(loss)})
            self.log_dict(metrics, batch_size=masked_data.shape[0])

        return loss

    def training_step(self, input_batch, idx):
        masked_data, original_data, embedding_x, embedding_y = input_batch
        return self._run(
            masked_data,
            original_data,
            embedding_x,
            embedding_y,
            determine_metrics=False,
            phase="train",
        )

    def validation_step(self, input_batch, idx):
        masked_data, original_data, embedding_x, embedding_y = input_batch
        return self._run(
            masked_data,
            original_data,
            embedding_x,
            embedding_y,
            determine_metrics=True,
            phase="val",
        )

    def impute(self, masked_data, original_data, time_marks_x, time_marks_y):
        self.eval()
        with torch.no_grad():
            mask = torch.isnan(masked_data)
            input_data = torch.nan_to_num(masked_data, nan=0.0)
            output, _ = self.forward(input_data, time_marks_x, mask=mask)

            result = original_data.clone()
            result[mask] = output[mask]

        return result
