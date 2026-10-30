# Data Model: 研究知识与可靠交接

设计状态：本期拟实现模型，新增字段尚未实现。字段/API 约束见 [ResearchUpdate 合同](contracts/research-update.md)、[CLI 合同](contracts/cli-output.md) 和 [派发协议](contracts/autoresearch-protocol.md)。

## 存储与增量兼容

继续使用 `.aimx/research/state.sqlite3` 的 `metadata/commits/events`：一个显式 ResearchUpdate 追加一个 commit 与按 sequence 排序的 events，在同一事务内更新 revision。新增投影从历史回放，不新增可独立编辑的实体表。ResearchUpdate、commit 与 storage metadata 的 schema_version 保持 1；直接扩展已有模型和事件目录，不维护两套解析/回放路径。compiler_version 仅作内部算法标记，不是 CLI 选择。

新读者读取旧记录并在内存填充 `source_activity_id=null`、`result_kind=unknown`、缺失科学范围等未知值，不改历史 payload。既有 update 必需字段不变，新增可选字段提供时严格校验；新 operation 校验自身必需字段。008 工作流的完整性检查独立于旧载荷兼容，不通过格式版本区分。所有实体保留 author、created_commit_id、created_revision；修订通过新事件或新 Finding 表达，时间为 UTC，排序依据 revision/ID，不依据读取时钟。

## Research Activity / Loop 来源登记

| 字段 | 类型与规则 |
| --- | --- |
| id | `act_...`，由现有 ID 机制生成；支持同 update `$local_id` 引用 |
| kind | `loop / analysis / follow_up` |
| title, objective | 非空字符串 |
| loop_id | 仅 loop，repo 内唯一；匹配 control 的 ID_PATTERN |
| related_loop_id | 仅 follow_up；引用本 repo 已登记 loop |
| source | 不可变 `{kind, location, note?}`，由作者说明登记来源 |
| report_locations | 位置字符串数组；只作导航，不自动升级为 EvidenceReference |
| artifact_locations | 位置字符串数组；与结构化已登记证据分开 |
| created_* / author | 标准来源字段 |

loop 的 checkpoint 固定在 `<repo>/.aimx/autoresearch/<loop_id>/state.json`；登记时验证该文件的 repo/ID 与活动一致，读取时失败仅报告 checkpoint 状态。analysis 无轮次；follow_up 不占原轮次。新增位置通过 `activity.artifact.add` 追加，不替换原来源。

Research State 对活动只保存身份/来源和知识归属。checkpoint 是 loop config、原标准、rounds_started、slot 状态、终止事实的唯一来源；Receipt 不把它们复制为另一份可变真相。旧 checkpoint 可显式登记 loop 来源，不修改 checkpoint 或自动给旧 annotation 归属。

## Research Receipt（只读视图）

| 字段 | 来源 |
| --- | --- |
| repo, state_location, revision | 已归一化 repo 和 sidecar 当前快照 |
| loop_id, source_activity_id | 选择的 loop 登记/显式 checkpoint |
| loop_status / checkpoint_status | checkpoint；不存在/不可读/不支持单独标识 |
| goal_status / goal_assessment | 原 criterion 的证据判断；`met/not_met/unknown` |
| direction_status | 相关 Frontier；与 loop 终止状态独立 |
| stop_reason, rounds_started, max_rounds | checkpoint 原事实；缺失时 null，不猜测 |
| counts / summaries | 本活动 Finding、annotation、Agenda；旧无归属计数独立 |
| persisted_receipts | 可核对的 key/commit_id/revision，不以产物存在代替 |
| external_artifacts / report_locations | checkpoint/活动导航，附已知可读情况 |
| pending_reconciliation | exact update、缺回执、指纹冲突、不可读来源等事项 |
| follow_up_activities | 独立来源活动列表，明确并非原轮次成果 |
| query_commands | 按实际 repo 和 ID 生成可复制、正确 shell 引号的入口 |

回执允许部分状态未知；状态库不可读时不能声称知识完整入库。合法空状态没有 `.aimx`；查询不得为了回执创建文件。Loop 原始 `running/blocked/interrupted/goal_met/budget_exhausted` 枚举保留，工作流服务失败映射到 blocked/interrupted + 原因，而非冒造已终止成功。

## Finding

保留原 `id/claim/evidence/epistemic_status/confidence/governance_status/history/created_*/author`。

| 新字段 | 验证规则 |
| --- | --- |
| source_activity_id | 既有 create 的可选扩展；008 工作流必填，引用本 repo 已有/同 update 活动；缺失未归属 |
| topics | 可选；提供时为非空、去重字符串数组；008 成果必填，确定性归一化匹配 |
| result_kind | 可选；提供时为 `improvement/degradation/insufficient_evidence/scope_boundary`，缺失 unknown |
| scope | 可选；提供时必含 `description` 非空、`dataset_scope_ids` 数组（可空）、`conditions` 字符串数组 |
| limitations | 可选字符串数组；008 成果必须明确，可为空但不代表已证明无局限 |
| source_annotation_ids | 显式提炼时非空 ID 列表；原记录必须存在，保留原 author/revision |

旧 create 规则仍可用，无证据 Finding 继续可读且显示缺项。声明 result_kind/scope/limitations 任一结果语义字段时，须一并提供这三项、topics、source_activity_id 及至少一条结构化证据；008 成果全部遵守这一完整性规则。Finding 的 claim、结果性质、scope、证据内容身份不原地改；新认识新建 Finding 加 refines/challenges/supersedes。`finding.assess` 仍可追加认知状态/置信度。`candidate/validated/contradicted` 与 `proposed/accepted/rejected` 沿用旧语义；无充分证据可保持 candidate，validated 也不等同整体目标 met。四类 result_kind 不是 execution status。

## Evidence Reference / Check / Review

| kind | 不可变身份 |
| --- | --- |
| aim_run | 既有 full `run_hash`；保留 role/note；增量允许证据类型、用途、来源活动，008 工作流补全，旧 Aim 引用不增必填字段。SDK 能力缺失与 run 缺失区分 |
| local_artifact | 原 `location`、`resolved_location`、`sha256`（64 位小写十六进制）、`size_bytes`（非负整数）、`evidence_type`、`purpose`、`source_activity_id`；role/note 可选 |

本地位置允许 repo 相对路径或绝对路径；相对路径只按 repo 根解析。登记读取普通文件计算内容身份并与声明比较，symlink 保留原位置与解析位置；不复制、不搬运、不执行文件、不抓 URL。读取前后 stat 与流式 digest 不一致即拒绝写入。登记后的位置/digest 不被读取改写。

证据在 Finding 中以规范内容身份+位置定位：审核事件携带 finding_id 与对应 ref 的 SHA-256 canonical JSON fingerprint，避免对错误引用更新审核。相同内容不同用途保留用途；context 可以按身份去重，但必须保留每个 finding 的用途映射。

| 状态轴 | 值与说明 |
| --- | --- |
| availability | `available/missing/unreadable/unsupported/unknown`；可读性与 SDK 能力 |
| identity_status | `matches/changed/not_checked/unknown`；Aim run 仅保证引用身份，不保证 run 可变内容的 digest |
| scientific_review | `not_reviewed/accepted/rejected/inconclusive`；显式作者/依据/范围 |

EvidenceCheck 是只读响应，返回 observed_location/hash/size、原引用、状态、错误原因；普通 context 读取仅带已知状态/not_checked，不哈希全量文件。明确 evidence check 不改变 revision。Review 使用 `evidence.review` 追加历史，要求 reviewer author、理由和支持依据；文件 matches 不会自动变成 accepted。

## Finding Relation

沿用 `rel_...`、source_finding_id、target_finding_id、type、active、reason、created_*、retracted_commit_id；可选增加 source_activity_id，008 工作流带明确来源。关系类型沿用 `supports/challenges/refines/supersedes/derived_from/tests/related_to`。

source 指向 target：例如后续限制 Finding `refines` 早期收益 Finding。必需修正闭包无向遍历 active challenges/refines/supersedes，包含全链与环，visited 保证终止。关系跨活动合法；任何 loop seed 的解释不能因 loop 过滤丢跨活动反证。retract 追加事件保留原关系历史。

## Context Selection（临时响应）

字段包括 schema/compiler version、repo/revision、objective/loop filter、budget unit/scope/limit/used、items、evidence_refs、selection_notes。items 是 `{kind,id,data}` 的紧凑解释投影，完整历史由 show/state 入口获取；不能删掉认识状态、范围、限制或重要反证以省预算。

selection_notes 至少有选择政策、每个选中组的 seed/理由、遗漏数量/按原因计数、历史关系限制、获取完整记录的命令。大量遗漏 ID 用计数和再查询入口表示，避免 omissions 本身撑爆预算。选择不依赖实时文件哈希/时钟。统一预算计实际 JSON 响应的 UTF-8 序列化字节数，含所有字段；完整必需知识组和最小 envelope 放不下则 exit 2，stdout 不含部分结论。

## Agenda / Experiment Contract

Agenda 保留既有 motivating_finding_ids/frontier_item_ids、status、priority、旧 controls/factors/constraints/success_criteria；`agenda.item.create` 可选增加 source_activity_id 与 experiment_contract。研究探索中的非实验任务可不带合同，但 skill 派发实验前合同必须完整。

严格合同包含 contract_version=1、question、interventions、fixed_controls、expected_evidence、disconfirmation、confounders、evaluation、data、resources、failure_policy。evaluation 保存项目 evaluator 命令（argv 数组）、协议版本/digest、metric 与 acceptance_rule、preflight report 身份；不在 Aimx 中实现统计公式。data 引用 DatasetScope，声明 exploratory 或 independent_confirmation 并绑定 candidate fingerprint；resources 是轮次外的计算预算声明。

合同冻结后不原地改；新计划使用新 Agenda/合同 ID。联合干预必须列出混杂，缺乏隔离证据不得声称机制已隔离。

## Dataset Scope / Observation Record

| 实体 | 关键字段 |
| --- | --- |
| DatasetScope `ds_...` | project_key、dataset_key、data_revision、description、overlap_keys（项目给出，非空）、coverage（`declared_complete/partial/unknown`）、initial_observation（`declared_unobserved/unknown`）、source_activity_id、source |
| Observation `obs_...` | dataset_scope_id、event_kind、occurred_at、source_activity_id、source、evidence、candidate_fingerprint（冻结/评分必需）、attempt_id（评分必需） |

event_kind 为 `scope_declared/candidate_frozen/first_scored/promoted_to_development`。评分对同 candidate/attempt 要有此前冻结事件；转开发与首次评分不可撤销以使数据“再次干净”。已有记录可补充后来的真实事件，但不能删除曾观察事实。实际 occurred_at 与入库 revision 都保留；迟到或冲突事件将独立性标为 unknown/violated，不靠改事件排序修复。

派发前仅当完整覆盖声明、明确未观察来源、冻结 candidate、无与该范围重叠的既有评分/开发/矛盾事件时可声明 independent_confirmation。首次评分可产生该已冻结候选的确认结果；评分后同窗口不能再次被称为未观察独立窗口。不同/未知 revision 或 overlap key 无法证明不重叠时为 unknown。检查仅证明记录一致，不能证明记录外无标签访问。

## Execution Outcome / Attempt

| 字段 | 规则 |
| --- | --- |
| id | `attempt_...` 或项目唯一 attempt ID，由 activity 内唯一性约束去重 |
| source_activity_id / agenda_item_id | 显式归属；loop source 包含 round/slot/native_task_id |
| retry_of | 可选旧 attempt ID；重试不覆盖旧记录，不授权重放 slot |
| execution_status | `succeeded/failed/interrupted/unknown` |
| artifact_status | `none/partial/complete/unknown` |
| evaluation_status | `not_run/passed/failed/inconclusive/unknown` |
| audit_status | `not_run/passed/failed/unknown` |
| evidence, summary | 保存已有产物和科学依据；summary 非空 |
| requested_settings / actual_settings | 配置来源与原生元数据分开；未知 actual 不仿造 |
| resources | `{kind,unit,planned,actual?,verification,enforcement,source?}` 数组 |

verification 为 `verified/unverified`，enforcement 为 `executor_enforced/declared_only/unknown`；verified 必须有执行器计数依据。多 attempt 已发生时全部记录，重复/超额显式报告；helper 只硬限制 rounds，不能保证 GPU/token 预算。Attempt 初次完整终结记录不可改；后续审计创建补充 Finding/Review 或新 outcome 来源，不覆盖原失败。执行失败而产物 complete、evaluation passed 是合法组合；总体 goal 独立判断。

## Preflight 与 Checkpoint 协议

Preflight 报告是一份冻结 JSON 产物，不是读取自动写入的状态。包含 aimx 入口/版本/合同能力、研究 state 可读性、必要 Aim 读取、每轮模型/effort 支持、项目 evaluator、failure_policy、每项 available/missing/unverified 及验证来源。Lead 显式引用产物身份再派发。

checkpoint 结构、config 规范化/fingerprint 算法及原 result 形状保持不变，无升级命令或双版本状态机。Preflight、合同、Attempt 四轴与 update digest 放在关联研究记录/产物中，由既有 artifacts/update_file/research_update 回执定位；新 loop 和旧 loop 使用同一机制。原 result.status 保留，缺新轴显示 unknown，不从执行成功推断科学成功。

Loop transitions 沿用 running ↔ blocked/interrupted；goal_met/budget_exhausted 终态不可重开。slot pending→reserved→started→finished，不退款、不替换模型、不从 finished 回退。继续旧循环前核对并补充关联合同/预检，缺项阻塞派发，不改写已完成事实或指纹配置。终态只能通过独立 follow_up 活动补证。

## 提交与交接一致性

1. Lead 保存确切 update 文件、canonical request fingerprint、原文件 digest 和稳定 client_update_id。
2. 显式 commit 原子追加知识；回执固定 commit_id/revision/created_ids。
3. checkpoint 引用回执，失败时保留 update/产物并报告待协调。
4. receipt 查询或精确重放按 key/指纹确定结果；同 key 不同 payload 拒绝，revision conflict 需刷新重新审议。
5. 结束/暂停/失败都读取 checkpoint 与知识生成同一 Receipt；证据保存、知识提交、目标达成分别表达。

这里没有跨 SQLite、JSON、外部文件的分布式事务，任何阶段失败都不能伪称交接完整。
