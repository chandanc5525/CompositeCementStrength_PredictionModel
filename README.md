# 🚀 Chandan-AIOps

[![PyPI version](https://img.shields.io/pypi/v/chandan-aiops.svg)](https://pypi.org/project/chandan-aiops/)
[![Python](https://img.shields.io/pypi/pyversions/chandan-aiops.svg)](https://pypi.org/project/chandan-aiops/)
[![License](https://img.shields.io/pypi/l/chandan-aiops.svg)](https://pypi.org/project/chandan-aiops/)

**Chandan-AIOps** is a lightweight, enterprise-ready MLOps project scaffolding tool designed to help Data Scientists and ML Engineers build production-grade machine learning systems with standardized architecture.

It accelerates the transition from experimentation to deployment by generating a clean, modular, scalable project structure.

---

## 📦 Installation

Install directly from PyPI:

```bash
pip install chandan-aiops
```
## Create Virtual Enviornment

```bash
python -m venv .venv
```

## Activate Virtual Environment

```bash
.venv\Scripts\activate
```
## Install Dependency Manager (uv)

```bash
pip install uv
```

## Initialize Project Dependency System

```bash
uv init
```

## Add Required ML & Deployment Libraries

```bash
uv add chandan-aiops numpy pandas scikit-learn ......
```

## Sync Dependencies

```bash
uv sync
```

## Generate Enterprise MLOps Structure using chandan-aiops

```bash
chandan-aiops create .
```
## Configure MLOPs file structure and then use Docker for containerization

```bash
docker build -t your-app-name .
docker run -p 8000:8000 your-app-name
```
