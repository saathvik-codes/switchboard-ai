"""AWS deployment: the classifier loaded from S3 (moto's in-memory AWS), the Lambda handler, and the SAM template."""
import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

import predict

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bucket(monkeypatch, tmp_path):
    for k, v in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "ap-south-1"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MODEL_BUCKET", "models")
    monkeypatch.setenv("MODEL_PREFIX", "switchboard/test")
    monkeypatch.setattr(predict.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(predict, "_clf", None)
    monkeypatch.setattr(predict, "_labels", None)
    with mock_aws():
        s3 = boto3.client("s3", region_name="ap-south-1")
        s3.create_bucket(Bucket="models", CreateBucketConfiguration={"LocationConstraint": "ap-south-1"})
        yield s3


def test_classifier_is_loaded_from_s3_when_a_bucket_is_configured(bucket, tmp_path):
    for name in predict.ARTIFACT_FILES:
        bucket.upload_file(str(predict.ARTIFACTS_DIR / name), "models", f"switchboard/test/{name}")
    clf, labels = predict._load()
    assert len(labels) == 77 and clf.n_features_in_ == 384
    assert (tmp_path / "switchboard-model" / "switchboard" / "test" / "intent_classifier.joblib").exists()


def test_missing_artifact_in_s3_is_an_error_not_a_silent_fallback(bucket):
    with pytest.raises(Exception):
        predict._load()                                   # nothing uploaded: refuse to serve rather than guess


def test_without_a_bucket_the_bundled_artifacts_are_used(monkeypatch):
    monkeypatch.delenv("MODEL_BUCKET", raising=False)
    assert predict._artifacts_dir() == predict.ARTIFACTS_DIR


def http_api_event(method: str, path: str) -> dict:
    """The event API Gateway (HTTP API, payload v2) sends to Lambda."""
    return {"version": "2.0", "routeKey": "$default", "rawPath": path, "rawQueryString": "",
            "headers": {"host": "api.example.com"}, "isBase64Encoded": False,
            "requestContext": {"http": {"method": method, "path": path, "protocol": "HTTP/1.1", "sourceIp": "1.2.3.4",
                                        "userAgent": "test"}, "stage": "$default", "requestId": "r1",
                               "routeKey": "$default", "accountId": "1", "apiId": "a", "domainName": "api.example.com",
                               "domainPrefix": "api", "time": "01/Jan/2026:00:00:00 +0000", "timeEpoch": 0}}


def test_lambda_handler_serves_the_api():
    import lambda_handler
    r = lambda_handler.handler(http_api_event("GET", "/health"), {})
    assert r["statusCode"] == 200 and json.loads(r["body"]) == {"status": "ok"}


def test_sam_template_is_valid_cloudformation():
    from cfnlint.api import lint_all
    problems = [m for m in lint_all((ROOT / "template.yaml").read_text(encoding="utf-8")) if m.rule.severity == "error"]
    assert problems == [], problems


def test_bucket_is_private_and_the_function_can_only_read_it():
    import yaml

    class Loader(yaml.SafeLoader):
        pass
    for tag in ("!Ref", "!Sub", "!GetAtt"):
        Loader.add_constructor(tag, lambda loader, node: loader.construct_scalar(node))
    t = yaml.load((ROOT / "template.yaml").read_text(encoding="utf-8"), Loader=Loader)
    block = t["Resources"]["ModelBucket"]["Properties"]["PublicAccessBlockConfiguration"]
    assert all(block.values())
    assert t["Resources"]["AgentFunction"]["Properties"]["Policies"] == [{"S3ReadPolicy": {"BucketName": "ModelBucket"}}]
