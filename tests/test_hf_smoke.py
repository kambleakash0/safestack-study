"""Real tiny-model generation through hf_local. Requires the [hf] extra; skipped otherwise.

Run with: uv run --extra hf pytest -m hf
"""

import pytest

pytestmark = pytest.mark.hf


def test_tiny_gpt2_generates_on_cpu():
    from safestack.config import DecodeParams, ModelSpec
    from safestack.model_gateway.base import GenerationRequest
    from safestack.model_gateway.hf_local import HFLocalGateway

    spec = ModelSpec(
        model_id="tiny_gpt2",
        backend="hf_local",
        checkpoint="sshleifer/tiny-gpt2",
        chat_template="none",
        device="cpu",
    )
    gateway = HFLocalGateway(spec)
    try:
        result = gateway.generate(
            GenerationRequest.from_prompt("Hello", DecodeParams(max_new_tokens=5))
        )
        assert isinstance(result.text, str)
        assert result.content_hash.startswith("sha256:")
        assert result.output_tokens is not None
    finally:
        gateway.close()


def test_hf_unknown_dtype_fails_loudly():
    import pytest

    from safestack.config import DecodeParams, ModelSpec
    from safestack.model_gateway.base import GenerationRequest
    from safestack.model_gateway.hf_local import HFLocalGateway

    spec = ModelSpec(
        model_id="x",
        backend="hf_local",
        checkpoint="sshleifer/tiny-gpt2",
        dtype="bogus",
        device="cpu",
    )
    gateway = HFLocalGateway(spec)
    with pytest.raises(ValueError):
        gateway.generate(GenerationRequest.from_prompt("hi", DecodeParams()))

def test_generate_batch_matches_single_sequence_and_cache_keys():
    # Batched generation must (a) give the SAME content_hash per prompt as single
    # generate() -- so the cache stays coherent across single/batched runs (resume,
    # mixing) -- and (b) give identical text where there is no left-padding confound
    # (batch-of-1; identical-length prompts). Heterogeneous-padding greedy equivalence
    # is validated on the production run (a documented ADR footnote): on a tiny random
    # model, batched-matmul float noise can flip a knife-edge argmax, so exact text
    # equality is asserted here only for the no-padding cases.
    from safestack.config import DecodeParams, ModelSpec
    from safestack.model_gateway.base import GenerationRequest
    from safestack.model_gateway.hf_local import HFLocalGateway

    spec = ModelSpec(
        model_id="tiny_gpt2",
        backend="hf_local",
        checkpoint="sshleifer/tiny-gpt2",
        chat_template="none",
        device="cpu",
    )
    gw = HFLocalGateway(spec)
    try:
        params = DecodeParams(max_new_tokens=12, seed=0)  # greedy (do_sample defaults to False)
        assert gw.generate_batch([]) == []  # empty batch is a no-op

        # (a) content_hash coherence across ALL prompts, incl. varied lengths (real left-padding)
        varied = ["Hello", "The quick brown fox jumps over the", "A B C"]
        vreqs = [GenerationRequest.from_prompt(p, params) for p in varied]
        vbatched = gw.generate_batch(vreqs)
        assert [r.content_hash for r in vbatched] == [gw.generate(r).content_hash for r in vreqs]
        assert len(vbatched) == 3 and all(isinstance(r.text, str) for r in vbatched)

        # (b) exact text equality where there is no padding confound
        one = GenerationRequest.from_prompt("Hello world", params)
        assert gw.generate_batch([one])[0].text == gw.generate(one).text  # batch-of-1 == single
        pair = [GenerationRequest.from_prompt("A B C", params) for _ in range(2)]  # no padding
        pb = [r.text for r in gw.generate_batch(pair)]
        assert pb[0] == pb[1] == gw.generate(pair[0]).text
    finally:
        gw.close()

def test_judge_score_batch_matches_single_sequence():
    # A HF-backed judge's score_batch must match per-item score() where there is no left-padding
    # confound (batch-of-1), and return one label per item in order for a varied batch. It reuses
    # the gateway's batched generation (verified above); parse functions are pure + unit-tested.
    from safestack.config import ModelSpec
    from safestack.eval.judges.helpfulness import HelpfulnessJudge

    spec = ModelSpec(
        model_id="tiny_gpt2",
        backend="hf_local",
        checkpoint="sshleifer/tiny-gpt2",
        chat_template="none",
        device="cpu",
    )
    judge = HelpfulnessJudge(spec)
    try:
        assert judge.score_batch([]) == []
        one = judge.score_batch([("what is 2+2", "4")])  # batch-of-1: no padding
        assert len(one) == 1 and one[0].label == judge.score("what is 2+2", "4").label
        varied = judge.score_batch([("q", "short"), ("a longer question", "a longer answer too")])
        assert len(varied) == 2 and all(hasattr(x, "label") for x in varied)
    finally:
        judge.close()
