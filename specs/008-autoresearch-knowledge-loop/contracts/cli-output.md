# CLI / Output Contract

这是 008 的拟实现合同；原接口见 [007 CLI 合同](../../007-research-state/contracts/cli-output.md)。只扩展已有 Aimx-owned `research/finding/lineage/frontier` 命令树；其它 root 继续 native Aim argv/stdout/stderr/exit 透传。

## 公共约定

repo 接受 repo 根或 `.aim` 目录，输出真实规范化根；不存在目录错误，不把未知路径当空状态。存在目录但 sidecar 不存在是合法 revision 0，不创建 `.aimx`。所有读取/帮助/dry-run/check 为只读；显式写只到 Aimx sidecar。

退出码沿用 0 成功（包括无匹配/合法空）、1 内部错误、2 参数/schema/证据/状态/预算错误、3 revision conflict。正常输出 stdout，预期错误 stderr：

```json
{"schema_version":1,"error":{"code":"invalid_input","message":"...","details":{}}}
```

读写 JSON 的 schema_version 均保持 1；读接口含 repo/revision（状态读取失败时 revision 为 null 并标 partial）。新增字段/事件增量扩展，旧输入必需字段和旧记录可读性保持。上下文 budget_used 统一计完整响应，行为变化详见 context 合同；compiler_version 只标算法，不需要用户选择。Human 与 JSON 传达同一事实，无交互 prompt。

## 所有层级帮助（FR-003）

```text
aimx research -h|--help
aimx research <state|update|context|agenda|next|overview|annotations|receipt|evidence> -h|--help
aimx research evidence check -h|--help
aimx finding -h|--help
aimx finding <ls|show|comment|accept|reject|assess> -h|--help
aimx lineage -h|--help
aimx lineage <show|link|retract> -h|--help
aimx frontier -h|--help
aimx frontier <show|lane-add|add|move|pause|retire> -h|--help
```

已识别层级的 help 在 repo、实体 ID、SDK/状态读取前返回 exit 0。用途、所有参数、示例、读写性质、退出码必须出现。缺少 positional ID 或 `--repo /不存在` 不妨碍帮助。未知子命令仍报 invalid_input；native Aim 子命令 help 不被全局截获。help 默认静态文本；`--json` 输出静态 `{schema_version:1,command,usage,description,mutation,options,examples,exit_codes}`，无 revision/repo 数据读取。

## 实际入口预检

复用现有 version、各层 help、state/read 与 update --dry-run 验证实际使用的 Aimx 入口，不新增版本协商命令。Skill 记录入口绝对路径、包版本与所需能力的实际调用结果；缺命令/文件证据能力时报告具体缺项，不根据版本号推断就绪。

## `research overview`

```text
aimx research overview [--repo PATH] [--loop LOOP_ID] [--json]
```

无 loop 时返回 repo 概览和已知 loop 列表，不默认选最新 loop。登记 loop 的列表项包括 `finding_count`、`annotation_count` 和 `finding_summaries`，以便无需先选 loop 就能看出历史 annotation/Finding 是否存在；未归属记录仍单独按 repo 总计展示，不猜测归属。来源为登记 Activity 及标准 `.aimx/autoresearch/*/state.json`；目录扫描只读，只发现标准 checkpoint，不能导入或猜测知识归属。多 loop 需明确选择；`--loop` 不存在则 exit 2 unknown_loop。仅登记活动但 checkpoint 缺失返回 partial，绝不伪造终止状态。

单 loop JSON：

```json
{
  "schema_version":1,
  "repo":"/study",
  "revision":16,
  "status":"complete",
  "loop_id":"study-1",
  "source_activity_id":"act_loop",
  "state_location":"/study/.aimx/research/state.sqlite3",
  "checkpoint_location":"/study/.aimx/autoresearch/study-1/state.json",
  "checkpoint_status":"readable",
  "loop_status":"budget_exhausted",
  "goal_status":"not_met",
  "direction_status":"active",
  "stop_reason":"max_rounds reached",
  "rounds_started":10,
  "max_rounds":10,
  "goal_assessment":[],
  "counts":{"findings":0,"annotations":20,"agenda_items":10},
  "unattributed":{"findings":0,"annotations":0},
  "summaries":[],
  "persisted_receipts":[{"client_update_id":"study-1-round-10","commit_id":"commit_16","revision":16}],
  "external_artifacts":[],
  "report_locations":["/reports/study.md"],
  "pending_reconciliation":[],
  "follow_up_activities":[],
  "query_commands":["aimx research annotations --repo /study --loop study-1 --json"]
}
```

`status=complete` 只表示视图来源齐全，不能表示 goal met。DB 不可读或 checkpoint 损坏时输出 `status=partial`、缺项 null、errors/pending_reconciliation，exit 2；保留可安全读取的产物导航，不把它们称为已提交。unknown repo 直接错误。Receipt 由同一视图实现，工作流在结束/暂停/阻塞/失败时必须输出对应快照；已提交回执可能落后最新 revision，二者分别展示。

## Finding 与历史 annotation

```text
aimx finding ls [--repo PATH] [--topic TEXT] [--loop LOOP_ID] [--json]
aimx finding show FINDING_ID [--repo PATH] [--json]
aimx research annotations [--repo PATH] [--topic TEXT] [--loop LOOP_ID] [--json]
```

ls 使用 topics、claim、scope 的确定性词法匹配；topic 未命中返回空 findings，不能混入 governance 加权项。loop 只返回该 loop Activity 来源的 seeds，follow_up 单独给关联入口；无归属记录不混入。返回 `findings/match_reasons/other_record_counts/unattributed/query_commands`，Human 空结果说明“没有匹配 Finding；另有 N 条 annotation…”并提供带真实 repo 路径的查询入口，不能声称没成果。无过滤时旧 ls 排序与原字段保留。

show 保留完整 Finding/history，展示相邻修正及 `related_findings` 的范围/来源/证据导航；跨活动修正明确标记。annotations 查询返回原 text/author/revision，不生成 Finding。提炼通过作者显式提交新 Finding 的 source_annotation_ids，不增加自动 NLP 转换命令。

## Evidence check

```text
aimx research evidence check --finding FINDING_ID [--repo PATH] [--json]
```

只读检查该 Finding 全部引用，结果含原 ref、availability、identity_status、scientific_review、观察身份与原因。身份状态使用 `matches/changed/not_checked/unknown`；可用性使用 `available/missing/unreadable/unsupported/unknown`。缺失/不可读/变化是成功诊断输出，exit 0，调用者读各项状态；未知 Finding/无法读 sidecar 为 exit 2。绝不更新 stored digest/revision 或因 matches 认可科学结论；文件路径不是 shell 命令。Aim SDK 不可用只标对应 ref unsupported，不能阻断纯文件证据检查。

## 提交回执查询

```text
aimx research receipt --client-update-id KEY [--fingerprint SHA256] [--repo PATH] [--json]
```

只读查已有 commit 的 request fingerprint 与 result_json：匹配返回 `{schema_version:1,status:"committed",revision,commit_id,client_update_id,created_ids,request_fingerprint}`。没有该 key 返回 exit 0 `status=not_found`（仅代表查询快照未见，不能保证并发写不会随后完成）。指纹不匹配 exit 2 idempotency_conflict；旧行没有足够协调信息 exit 2 receipt_unverifiable。不得初始化 DB 或新建 commit。

## Existing reads / writes

research state/update/agenda/next 和 Finding/Lineage/Frontier 原写命令保留。state 新增 activities/dataset_scopes/observations/outcomes/evidence_reviews 投影，旧实体继续可读；读取不提升 schema、不转换 annotation。

```text
aimx research update --repo PATH (--stdin | --file UPDATE_JSON) [--dry-run] [--json]
```

沿用现有 schema_version=1 输入，增量合同见 [research-update.md](research-update.md)。所有 operation 原子校验和提交；client_update_id 保持可选，008 工作流必须提供以支持去重/协调，旧载荷不加此门槛。成功回执沿用原字段；同 key/相同 canonical 请求返回原回执且 `idempotent:true`。返回 revision conflict 后不得默默改 base/重提。dry-run 不创建目录或可见 ID，不证明尚未发生过同 key commit。

context 命令详见 [context.md](context.md)，skill 使用同一命令和预算规则，无版本参数。

## 新错误码

除 007 既有错误码外：`unknown_loop`、`checkpoint_unreadable`、`artifact_changed_during_read`、`artifact_identity_mismatch`、`receipt_unverifiable`、`incomplete_experiment_contract`、`independence_claim_rejected`。预检由 skill 产物记录 missing/unverified，不伪装为 Aimx 已提供执行能力。
