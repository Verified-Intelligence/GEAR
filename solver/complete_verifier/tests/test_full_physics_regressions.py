"""Regression tests for wide full-physics ONNX models."""

from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch

import arguments

arguments.Config.parse_config(args=[], verbose=False)


def test_scatter_elements_add_broadcasts_batch_and_backpropagates():
    from load_model import _ScatterElementsAdd

    data = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    indices = torch.tensor([[1, 1, 2], [0, 3, 3]])
    updates = torch.tensor(
        [[10.0, 20.0, 30.0], [5.0, 7.0, 11.0]], requires_grad=True)

    actual = _ScatterElementsAdd.apply(data, indices, updates, 1)

    torch.testing.assert_close(
        actual,
        torch.tensor([[1.0, 32.0, 33.0, 4.0], [6.0, 2.0, 3.0, 22.0]]),
    )
    actual.sum().backward()
    torch.testing.assert_close(data.grad, torch.full_like(data, 2.0))
    torch.testing.assert_close(updates.grad, torch.ones_like(updates))


def test_load_model_accepts_scatter_elements_reduction_add(tmp_path):
    from load_model import load_model_onnx

    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [1, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    indices = numpy_helper.from_array(
        np.array([[1, 1, 2]], dtype=np.int64), name="indices")
    updates = numpy_helper.from_array(
        np.array([[10.0, 20.0, 30.0]], dtype=np.float32), name="updates")
    node = helper.make_node(
        "ScatterElements", ["data", "indices", "updates"], ["output"],
        axis=1, reduction="add")
    graph = helper.make_graph(
        [node], "scatter_elements_add", [data], [output],
        initializer=[indices, updates])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    path = tmp_path / "scatter_elements_add.onnx"
    onnx.save(model, path)

    converted, shape = load_model_onnx(
        str(path), x=torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    assert shape == (4,)
    torch.testing.assert_close(
        converted(torch.tensor([[1.0, 2.0, 3.0, 4.0]])),
        torch.tensor([[1.0, 32.0, 33.0, 4.0]]),
    )


def test_wide_tensor_storage_caps_initial_domain_allocation():
    import tensor_storage

    arguments.Config["bab"]["tree_traversal"] = "depth_first"
    with patch.object(tensor_storage, "StackTensorStorage") as storage_cls:
        tensor_storage.get_tensor_storage(
            (0, 15367, 15367), initial_size=1024,
            dtype=torch.float64, device="cpu", concat_dim=0)

    assert storage_cls.call_args.args[1] == 4
