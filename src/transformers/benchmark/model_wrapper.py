import os
import argparse
import pickle
import logging

import torch
import numpy as np


CPUDEV = torch.device("cpu")

class ModelSerializer:
    """Wrapper class around a PT model that records losses after each training iteration and writes them to file
    """
    def __init__(self, model, fname, num_iterations=20, loss_fn=None):
        """
        Params:
            model: Underlying PT model
            fname: Name of file in which to record losses
            num_iterations: Number of loss records to keep.
                            Iterations exceeding this number will be ignored
            loss_fn: Optional loss function if needed
        """
        self.model = model
        self.losses = []
        self.fname = fname.replace("/", "_")
        self.num_iterations = num_iterations

    def __getattr__(self, attr):
        # forward other attributes to underlying model to preserve user interface
        return self.model.__getattribute__(attr)

    def __del__(self):
        self._export_losses(self.fname)

    def __call__(self, *args, **kwargs):
        loss_fn, target = None, None
        if "loss_fn" in kwargs:
            assert "loss_fn" in kwargs
            assert "target" in kwargs
            loss_fn = kwargs.pop("loss_fn")
            target = kwargs.pop("target")
        model_out = self.model(*args, **kwargs)
        if loss_fn:
            loss = loss_fn(model_out, target)
            if len(self.losses) < self.num_iterations:
                self.losses.append(loss.detach().to(CPUDEV))
            return model_out, loss
        else:
            # assume HF model which stores loss as first element
            loss = model_out[0]
            if len(self.losses) < self.num_iterations:
                self.losses.append(loss.detach().to(CPUDEV))
            return model_out

    def _export_losses(self, fname):
        if self.losses:
            with open(fname, "wb") as f:
                pickle.dump(self.losses, f)


class SerCompare:
    def __init__(self, root_dir, model_name):
        model_name = model_name.replace("/", "_")
        xla_fname = os.path.join(root_dir, f"{model_name}_xla.pickle")
        native_fname = os.path.join(root_dir, f"{model_name}_native.pickle")
        self.xla_losses = self.import_losses(xla_fname)
        self.native_losses = self.import_losses(native_fname)

    def import_losses(self, fname):
        with open(fname, "rb") as f:
            return pickle.load(f)

    def validate(self):
        logging.debug("Native : XLA")
        minlen = min(len(self.xla_losses), len(self.native_losses))
        passed = True
        for it, (xla_out, native_out) in enumerate(zip(self.xla_losses[:minlen], self.native_losses[:minlen])):
            logging.debug(f"{native_out} : {xla_out}")
            # Increase relative error threshold linearly every 5 iterations
            # TODO: Exponential increases may be more appropriate to match
            #   propagation of precision loss
            if not np.allclose(xla_out, native_out, rtol=((it//5+1)*1e-02), atol=1e-03):
                print(f"Accuracy check failed on epoch {it}")
                print(f"xla: {xla_out}")
                print(f"native: {native_out}")
                print(f"Relative error = {abs((native_out-xla_out) / native_out)}")
                passed = False
        print(f"Accuracy check {'passed' if passed else 'failed'}!")
        return passed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    data_dir = args.datadir
    model_name = args.model
    compare = SerCompare(data_dir, model_name)
    compare.validate()


if __name__ == "__main__":
    main()
