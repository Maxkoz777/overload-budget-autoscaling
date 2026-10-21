"""Minimal Kubernetes API client for in-cluster use (service-account token) or `kubectl proxy`."""
from __future__ import annotations

import json
import os
import ssl
import urllib.request

SA = "/var/run/secrets/kubernetes.io/serviceaccount"


class Kube:
    def __init__(self, proxy: str | None = None):
        if proxy:
            self.base, self.headers, self.ctx = proxy.rstrip("/"), {}, None
        else:
            self.base = f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:{os.environ['KUBERNETES_SERVICE_PORT']}"
            self.headers = {"Authorization": "Bearer " + open(f"{SA}/token").read().strip()}
            self.ctx = ssl.create_default_context(cafile=f"{SA}/ca.crt")

    def request(self, method: str, path: str, body: dict | None = None, content_type: str = "application/json"):
        data = json.dumps(body).encode() if body is not None else None
        headers = dict(self.headers)
        if data is not None:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=5, context=self.ctx) as r:
            return json.load(r)

    def get(self, path: str):
        return self.request("GET", path)

    def scale(self, namespace: str, deployment: str, replicas: int):
        return self.request("PATCH", f"/apis/apps/v1/namespaces/{namespace}/deployments/{deployment}/scale",
                            {"spec": {"replicas": int(replicas)}}, "application/merge-patch+json")
