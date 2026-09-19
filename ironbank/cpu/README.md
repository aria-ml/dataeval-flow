# DataEval Workflows (CPU)

DataEval Workflows container - CPU variant

* **Upstream Repository**: [https://github.com/aria-ml/dataeval-flow](https://github.com/aria-ml/dataeval-flow)
* **Base Image**: `ironbank/redhat/ubi/ubi9-minimal:9.5`
* **DISA STIG / SRG Compliance**:
  * Runs as unprivileged user `dataeval` with numeric UID/GID `10001:10001`
  * SetUID/SetGID bits stripped
  * Zero build-time network access (hermetic installation via offline wheelhouse)
  * Built-in `HEALTHCHECK` command
  * Minimal attack surface (Red Hat UBI Minimal base)

---

## Volume Mounts

| Mount Path | Description | Access Mode |
|---|---|---|
| `/dataeval` | Input datasets, models, and configuration files | Read-Only |
| `/output` | Evaluation results, artifacts, and generated reports | Read-Write |
| `/cache` | Optional disk computation cache | Read-Write |

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DATAEVAL_DATA` | Container input directory | `/dataeval` |
| `DATAEVAL_OUTPUT` | Container output directory | `/output` |
| `DATAEVAL_CACHE` | Cache directory (auto-enabled if mounted) | `/cache` |
| `CONTAINER_MODE` | Runtime acceleration mode (`cpu` or `gpu`) | `cpu` |

---

## Running the Container

### Usage

```bash
docker run --rm \
  --mount type=bind,source=/path/to/data,target=/dataeval,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  registry1.dso.mil/ironbank/dataeval/dataeval-flow-cpu:0.2.2
```

### Interactive Help

```bash
docker run --rm registry1.dso.mil/ironbank/dataeval/dataeval-flow-cpu:0.2.2 --help
```

---

## License

This project is licensed under the [MIT License](LICENSE).
