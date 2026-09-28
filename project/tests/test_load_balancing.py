"""Configuration contract checks only; never start containers or load models."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("docker"), "Docker CLI needed for Compose parsing")
class ComposeLoadBalancingTests(unittest.TestCase):
    @staticmethod
    def configuration(overrides=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("FASHION_")}
        env.update(overrides or {})
        result = subprocess.run(["docker", "compose", "-f", str(ROOT / "docker-compose.yml"),
                                 "config", "--format", "json"], cwd=ROOT, env=env,
                                capture_output=True, text=True, check=True, timeout=30)
        return json.loads(result.stdout)["services"]

    def test_only_gateway_publishes_api_and_frontend_uses_gateway(self):
        services = self.configuration()
        self.assertEqual(set(services), {"frontend", "nginx", "backend", "backend_2"})
        self.assertFalse(services["backend"].get("ports"))
        self.assertFalse(services["backend_2"].get("ports"))
        self.assertEqual(str(services["nginx"]["ports"][0]["published"]), "8000")
        self.assertEqual(services["nginx"]["ports"][0]["target"], 80)
        self.assertEqual(services["frontend"]["environment"]["VITE_API_PROXY_TARGET"], "http://nginx:80")
        for name in ("backend", "backend_2"):
            self.assertEqual(services["nginx"]["depends_on"][name]["condition"], "service_started")

    def test_instances_share_artifacts_and_have_separate_logs(self):
        services = self.configuration()
        first, second = services["backend"], services["backend_2"]
        for field in ("build", "image", "healthcheck", "volumes", "networks"):
            self.assertEqual(first[field], second[field])
        for name in ("CLIP_MODEL_PATH", "CLIP_EMBEDDING_DIR", "HF_HOME"):
            self.assertEqual(first["environment"][name], second["environment"][name])
        self.assertNotEqual(first["environment"]["RECOMMENDATION_LOG_NAME"],
                            second["environment"]["RECOMMENDATION_LOG_NAME"])

    def test_web_build_and_bind_paths_exist_after_reorganization(self):
        services = self.configuration()
        for name in ("backend", "backend_2", "frontend"):
            build = services[name]["build"]
            context = Path(build["context"])
            self.assertTrue((context / build["dockerfile"]).is_file())
        for name, target, relative in (
            ("backend", "/app/web/backend", "web/backend"),
            ("backend_2", "/app/web/backend", "web/backend"),
            ("frontend", "/app", "web/frontend"),
            ("nginx", "/etc/nginx/conf.d/default.conf", "web/nginx/default.conf"),
        ):
            mount = next(m for m in services[name]["volumes"] if m["target"] == target)
            self.assertEqual(Path(mount["source"]), ROOT / relative)
            self.assertTrue(Path(mount["source"]).exists())
        dockerfile = (ROOT / "web/backend/Dockerfile").read_text(encoding="utf-8")
        self.assertIn('"web.backend.main:app"', dockerfile)
        for line in dockerfile.splitlines():
            if line.startswith("COPY "):
                self.assertTrue((ROOT / line.split()[1]).exists(), line)

    def test_custom_catalog_mounts_apply_to_both_instances(self):
        variables = {"FASHION_CATALOG_DIR": (ROOT / "example experiment" / "data").as_posix(),
                     "FASHION_IMAGE_DIR": (ROOT / "example experiment" / "images").as_posix(),
                     "FASHION_EMBEDDINGS_DIR": (ROOT / "example experiment" / "embeddings").as_posix(),
                     "FASHION_CHECKPOINT_DIR": (ROOT / "example experiment" / "best").as_posix()}
        services = self.configuration(variables)
        targets = dict(zip(("/app/data/processed", "/app/data/data", "/app/embeddings_finetuned",
                            "/app/runs/clip_finetune_3epochs_local/best"), variables.values()))
        for name in ("backend", "backend_2"):
            mounts = {m["target"]: m for m in services[name]["volumes"]}
            for target, source in targets.items():
                self.assertEqual(Path(mounts[target]["source"]), Path(source))
                self.assertTrue(mounts[target]["read_only"])


class NginxConfigurationContractTests(unittest.TestCase):
    def test_dns_balancing_and_bounded_retry_contract(self):
        config = (ROOT / "web" / "nginx" / "default.conf").read_text(encoding="utf-8")
        for directive in ("resolver 127.0.0.11", "zone fashion_api 64k;", "least_conn;",
                          "server backend:8000 resolve", "server backend_2:8000 resolve",
                          "proxy_next_upstream_tries 2;", "client_max_body_size 6m;",
                          "proxy_request_buffering on;"):
            self.assertIn(directive, config)
        # Guard against accidentally retrying future POST write endpoints globally.
        lines = [line.strip() for line in config.splitlines() if line.strip().startswith("proxy_next_upstream ")]
        self.assertEqual(sum("non_idempotent" in line for line in lines), 2)
        for route in ("image", "text"):
            block = config.split(f"location = /api/recommendations/{route} {{", 1)[1].split("}", 1)[0]
            self.assertIn("non_idempotent", block)


if __name__ == "__main__":
    unittest.main()
