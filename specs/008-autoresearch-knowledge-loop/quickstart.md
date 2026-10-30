# Quickstart: 008 验证指南

本指南用于 008 实现后的验收。所有新写入都在临时 repo；真实 `data/.aim` 仅用于只读 smoke check，外部实验一律用假宿主/evaluator，不启动 LLM 或 GPU。

## 环境和既有基线

从仓库根执行；uv 管理 `.venv`，优先 Python 3.12：

```bash
uv sync --python 3.12
uv run python --version
uv run pytest tests/contract/test_research_contract.py tests/contract/test_autoresearch_protocol.py tests/unit/test_research_store.py tests/unit/test_research_context.py tests/unit/test_autoresearch_control.py tests/integration/test_autoresearch_skill_loop.py tests/integration/test_passthrough_behavior.py
uv run aim runs --repo data/.aim ls
```

预期既有测试通过，native Aim run 列表读取成功。没有 Aim SDK 的隔离测试环境仍须能读纯 sidecar/文件证据并看帮助，不能自动安装/迁移来掩盖 capability 缺项。

## 创建可移植文件证据样例

以下仅生成验证输入，不包含实现代码或完整测试套件。临时 repo 不需造 Aim run：

```bash
export AIMX_VALIDATION_REPO="$(mktemp -d)"
uv run python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ['AIMX_VALIDATION_REPO'])
artifact = root / 'results' / 'evaluation.json'
artifact.parent.mkdir()
artifact.write_text(json.dumps({'delta': 0.01, 'interval': [-0.02, 0.04]}, sort_keys=True) + '\n')
content = artifact.read_bytes()
update = {
    'schema_version': 1,
    'base_revision': 0,
    'client_update_id': 'portable-analysis-result-1',
    'author': {'kind': 'human', 'name': 'validation-lead'},
    'operations': [
        {
            'op': 'activity.create', 'local_id': 'activity', 'kind': 'analysis',
            'title': '可移植评估', 'objective': '判断候选收益证据',
            'source': {'kind': 'validation_fixture', 'location': 'results/evaluation.json'}
        },
        {
            'op': 'finding.create', 'local_id': 'finding',
            'claim': '估计区间跨零，当前证据不能证明收益改善',
            'source_activity_id': '$activity', 'topics': ['候选收益'],
            'result_kind': 'insufficient_evidence',
            'scope': {'description': '固定候选的样例窗口', 'dataset_scope_ids': [], 'conditions': ['同一 evaluator']},
            'limitations': ['未包含独立重复验证'],
            'epistemic_status': 'candidate', 'confidence': 'low',
            'evidence': [{
                'kind': 'local_artifact', 'location': 'results/evaluation.json',
                'sha256': hashlib.sha256(content).hexdigest(), 'size_bytes': len(content),
                'evidence_type': 'evaluation_report', 'purpose': '说明估计区间跨零',
                'source_activity_id': '$activity'
            }]
        }
    ]
}
(root / 'update.json').write_text(json.dumps(update, ensure_ascii=False, indent=2) + '\n')
print(root)
PY
```

## 文件 Finding、检查和提交协调（新增）

```bash
uv run aimx research update --repo "$AIMX_VALIDATION_REPO" --file "$AIMX_VALIDATION_REPO/update.json" --dry-run --json
```

预期 status=valid、base_revision=0；此时临时 repo 仍没有 `.aimx`。然后显式提交：

```bash
uv run aimx research update --repo "$AIMX_VALIDATION_REPO" --file "$AIMX_VALIDATION_REPO/update.json" --json > "$AIMX_VALIDATION_REPO/commit-receipt.json"
export AIMX_VALIDATION_FINDING="$(uv run python -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["AIMX_VALIDATION_REPO"])/"commit-receipt.json").read_text())["created_ids"]["finding"])')"
uv run aimx finding ls --repo "$AIMX_VALIDATION_REPO" --topic '候选收益' --json
uv run aimx finding show "$AIMX_VALIDATION_FINDING" --repo "$AIMX_VALIDATION_REPO" --json
uv run aimx research evidence check --finding "$AIMX_VALIDATION_FINDING" --repo "$AIMX_VALIDATION_REPO" --json
uv run aimx research receipt --client-update-id portable-analysis-result-1 --repo "$AIMX_VALIDATION_REPO" --json
uv run aimx research update --repo "$AIMX_VALIDATION_REPO" --file "$AIMX_VALIDATION_REPO/update.json" --json
```

预期完整来源/范围/限制/证据可追溯，identity matches、scientific_review not_reviewed；重复提交返回相同 commit_id/revision/created_ids 且 idempotent=true，不再创建 Finding。receipt 带同一 request fingerprint；fingerprint 不匹配返回 idempotency_conflict。`not_found` 仅表示查询快照未见该 key，并不保证另一个 writer 稍后不会提交。

在临时目录修改文件后再次 evidence check：

```bash
uv run python - <<'PY'
import os
from pathlib import Path
p = Path(os.environ['AIMX_VALIDATION_REPO']) / 'results/evaluation.json'
p.write_text('{"delta":999}\n')
PY
uv run aimx research evidence check --finding "$AIMX_VALIDATION_FINDING" --repo "$AIMX_VALIDATION_REPO" --json
```

预期 availability=available、identity_status=changed，原 stored digest/claim/revision 不变；文件删除为 availability=missing，权限错误为 unreadable，不能把 changed 写成科学 rejected。不可读 fixture 应用模拟 IO 错误保证 root/不同平台也稳定。

## 上下文预算和复现（统一规则）

```bash
uv run aimx research context --repo "$AIMX_VALIDATION_REPO" --objective '候选收益' --budget 18000 --json > "$AIMX_VALIDATION_REPO/context.json"
uv run python - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ['AIMX_VALIDATION_REPO']) / 'context.json'
raw = p.read_bytes()
value = json.loads(raw)
assert value['schema_version'] == 1
assert len(raw) == value['budget_used'] <= value['budget_limit']
assert value['evidence_refs']
print('response byte budget verified:', len(raw))
PY
uv run aimx research context --repo "$AIMX_VALIDATION_REPO" --objective '候选收益' --budget 1 --json
```

预算 1 预期 exit 2，stdout 无部分知识，stderr 明确预算不足且包含 required_bytes 与完整查询入口。完整十轮 fixture 需构造早期增益、多跳审计修正、最终结论和 20 条 annotation；18,000 bytes 含最终解释，N/N-1 要么完整闭包要么明确不足。再测全未匹配 topic、中文匹配、环、跨活动修正、annotations-only 最新相关结论、大量遗漏；固定输入/证据状态重复 JSON 字节一致。JSON 响应序列为 compact sorted JSON + LF，整个 UTF-8 响应计入预算；Human 输出显示 response_json scope。具体选择规则见 [context 合同](contracts/context.md)。

## 帮助和只读边界

对 [CLI 合同](contracts/cli-output.md) 中每个已拥有的父/叶路径枚举两种 help：

```bash
uv run aimx research overview --repo /nonexistent/aimx-validation --help
uv run aimx finding show --repo /nonexistent/aimx-validation -h
uv run aimx research evidence check --help --json
```

预期 exit 0、用途/参数/示例/读写/退出码齐全，无 repo/ID 要求；测试用 monkeypatch 禁止 repo/SDK/store 读取并验证未调用。未知子命令仍 exit 2，native Aim help 透传保留。缺少 Python Aim 包时，静态 help、纯 state 读取及本地文件证据仍可用；Aim run 引用单独报告 SDK 能力缺失。

用目录树/hash 快照比较 read/overview/help/context/check/dry-run 前后：不创建 `.aimx`，不改 sidecar/checkpoint/`.aim`。允许 OS 访问时间变化，不把 atime 当业务写入。

## Loop、follow-up 和协议验收

终态后补证用新的 `follow_up` Activity、文件证据 Finding 和指向旧 Finding 的 `refines`/`challenges`/`supports` 关系。主题查询和 `context --loop` 应同时展示原来结论与后续范围；原 checkpoint bytes、轮数、标准、assessment 不变。稳定 update key 的 receipt 快照 `not_found` 不能授权重放已 reserved 的宿主槽位；只可精确重放保存的 ResearchUpdate。receipt fingerprint 冲突后刷新状态并用新 key 审重，不重开 loop。

无需 LLM/GPU，扩充既有 `tests/integration/test_autoresearch_skill_loop.py` 的假宿主/evaluator，临时目录建立标准 checkpoint：

1. 构造成功、未达标、阻塞、仅 annotation、多循环至少五种 Receipt。按最多三次查看找到实际 repo、goal/status、成果和证据入口；计时验收两分钟内。旧无归属记录明确单列，非当前 loop 不混入。
2. 两轮模拟流程验证合同→预检→begin-round→reserved→started→finished→update→receipt。项目评分返回协议/预测身份和资源日志，不做真实训练。
3. 缺命令、旧 Aimx、Aim SDK 缺失但文件能力可用、模型不支持、evaluator 预检失败各有具体 capability 结果；required 缺项不 spawn，不能标 ready。
4. 首次评分、开发转换、未知覆盖、重叠范围、多候选重复评分、冻结次序冲突：所有错误 independent_confirmation 声明被拒绝；合法 exploratory 不因此失败。预检必须引用文件身份，verified resource 必须有 actual/source，开发转换绑定冻结候选。保留首次事件 ID 与记录外观察限制。
5. 至少五种故障：服务失败但完整预测、部分产物、预检失败、多 attempt、commit 成功回执中断。检查四状态轴、原 evidence、资源未核验和 pending_reconciliation。
6. 原终态 10/10 loop 新增 follow_up Finding/refines 后，主题 show/context 呈现局部收益与后续范围；原 checkpoint bytes、轮数、标准和终止 assessment 全不变。
7. 未声明策略止后续派发；continue_independent 只有宿主预验证时允许。禁止槽位重放/替换模型/退款/扩轮；旧 checkpoint 可直接读取，配置指纹保持原算法；新预检/合同/attempt 通过研究记录关联，不改写 checkpoint；继续旧循环前报告具体缺项。
8. 原 exact update/key 重放取回提交；同 key 不同 payload idempotency_conflict，revision conflict 不盲重试；旧无法核验回执明确错误。旧 payload 缺全部新增字段仍有效，新字段提供时严格校验；旧请求指纹不因补默认字段改变。现有 config fingerprint、finished.result 形状和终态读取不变。未匹配查询结果不伪装成命中。

## 实施后的验收运行

使用现有测试文件扩展行为验收；需要新 fixture 时放 `tests/fixtures/research_knowledge_loop/`（拟新增），不依赖 spec 中私有 ICL 路径。下列测试文件现在存在，新增断言由实施阶段补充：

```bash
uv run pytest tests/unit/test_research_validation.py tests/unit/test_research_replay.py tests/unit/test_research_evidence.py tests/unit/test_research_context.py tests/unit/test_research_store.py tests/unit/test_autoresearch_control.py tests/unit/test_research_scale.py
uv run pytest tests/contract/test_research_schema.py tests/contract/test_research_contract.py tests/contract/test_autoresearch_protocol.py tests/integration/test_research_commands.py tests/integration/test_autoresearch_skill_loop.py tests/integration/test_passthrough_behavior.py
```

不以测试文件名存在代替新增场景覆盖。最终验收报告列出 SC-001 至 SC-008 的 fixture/结果、FR 覆盖，以及所有未接入的项目执行器能力。规划阶段验证只检查文档/例子一致性，不宣称上述新行为已经实现。
