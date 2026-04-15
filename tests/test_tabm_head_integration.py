import torch

from pytorch_tabnet.tab_network import TabNetNoEmbeddings


def test_be_head_preserves_outer_output_shape():
    batch_size = 8
    input_dim = 12
    output_dim = 3
    x = torch.randn(batch_size, input_dim)

    baseline = TabNetNoEmbeddings(
        input_dim=input_dim,
        output_dim=output_dim,
        n_steps=2,
        n_d=4,
        n_a=4,
    )
    be_head = TabNetNoEmbeddings(
        input_dim=input_dim,
        output_dim=output_dim,
        n_steps=2,
        n_d=4,
        n_a=4,
        be_head_k=4,
        be_head_scaling_init="ones",
    )

    baseline_out, _ = baseline(x)
    be_out, _ = be_head(x)

    assert baseline_out.shape == (batch_size, output_dim)
    assert be_out.shape == baseline_out.shape


def test_be_head_ensemble_view_expands_batch_representations():
    batch_size = 5
    input_dim = 10
    output_dim = 2
    be_k = 3
    x = torch.randn(batch_size, input_dim)

    model = TabNetNoEmbeddings(
        input_dim=input_dim,
        output_dim=output_dim,
        n_steps=2,
        n_d=4,
        n_a=4,
        be_head_k=be_k,
        be_head_scaling_init="ones",
    )

    steps_output, _ = model.encoder(x)
    res = torch.sum(torch.stack(steps_output, dim=0), dim=0)
    expanded = model.head_ensemble_view(res)

    assert expanded.shape == (batch_size, be_k, model.n_d)


def test_be_head_train_eval_forward_sanity():
    batch_size = 7
    input_dim = 14
    output_dim = 1
    x = torch.randn(batch_size, input_dim)

    model = TabNetNoEmbeddings(
        input_dim=input_dim,
        output_dim=output_dim,
        n_steps=2,
        n_d=4,
        n_a=4,
        be_head_k=2,
        be_head_scaling_init="random-signs",
    )

    model.train()
    train_out, _ = model(x)
    model.eval()
    eval_out, _ = model(x)

    assert train_out.shape == (batch_size, output_dim)
    assert eval_out.shape == (batch_size, output_dim)


def test_be_head_does_not_change_mask_shapes():
    batch_size = 6
    input_dim = 9
    x = torch.randn(batch_size, input_dim)

    baseline = TabNetNoEmbeddings(
        input_dim=input_dim,
        output_dim=2,
        n_steps=2,
        n_d=4,
        n_a=4,
    )
    be_head = TabNetNoEmbeddings(
        input_dim=input_dim,
        output_dim=2,
        n_steps=2,
        n_d=4,
        n_a=4,
        be_head_k=3,
        be_head_scaling_init="ones",
    )

    baseline_explain, baseline_masks = baseline.forward_masks(x)
    be_explain, be_masks = be_head.forward_masks(x)

    assert be_explain.shape == baseline_explain.shape
    assert be_masks.keys() == baseline_masks.keys()
    for step in baseline_masks.keys():
        assert be_masks[step].shape == baseline_masks[step].shape
