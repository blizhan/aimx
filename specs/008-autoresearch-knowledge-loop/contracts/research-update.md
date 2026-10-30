# ResearchUpdate 增量扩展合同

本文件为本期规范性 CLI 输入合同；采用 operation 表和字段校验规则，沿用 [现有 JSON Schema](../../007-research-state/contracts/research-update.schema.json) 的顶层形状。实施时同步更新该现有 schema、Python 校验器与合同测试，逐 operation 校验，不把未知键静默丢弃；不另建平行版本 schema。

## Envelope / reference rules

```json
{
  "schema_version":1,
  "base_revision":0,
  "client_update_id":"analysis-1-result",
  "author":{"kind":"human","name":"lead"},
  "message":"保存文件支持的证据不足结论",
  "operations":[]
}
```

required 继续为 schema_version/base_revision/author/operations；schema_version 保持 1，base_revision 非负整数（bool 非整数），author.kind 为 agent/human/system，name 非空，operations 至少 1 项。client_update_id、message 保持可选字符串；提供 key 时必须非空。未知顶层字段拒绝。直接在现有 operation 中增加明确允许的可选字段和新 operation，不按版本分派；旧载荷原样有效。Skill 的研究提交必须带稳定 client_update_id，以获得可协调的交接保证。

`local_id` 可选，沿用 `[A-Za-z][A-Za-z0-9_-]*`；同 update 唯一，`$local_id` 只能引用前序声明实体。持久 ID 必须存在且种类正确。新增活动、观察、outcome 等 operation 按下表要求来源。既有 operation 的 source_activity_id 为可选扩展，缺失即未归属；008 工作流生成的记录必须带来源。关系跨活动允许；来源必须真实声明，不自动猜。

## Operation catalogue

现有 `finding.create/assess/governance`、`relation.create/retract`、`annotation.create`、`frontier.lane.create/rename/retire`、`frontier.item.create/move/set_priority/set_status/update`、`agenda.item.create/transition/attach_evidence/set_priority` 全部保留原必需字段、状态和转换。下表中既有 operation 的新字段为可选；一旦提供，严格校验。评估/状态转换/撤关系可保存本次 source_activity_id，事件 provenance 不抹掉原实体来源。新增 operation 的必需字段不受此兼容规则放宽。

| 新/扩展 operation | 必需字段（除 op） | 可选字段 / 约束 |
| --- | --- | --- |
| activity.create | kind,title,objective,source | local_id；loop_id（loop 必需）；related_loop_id（follow_up 必需）；report_locations/artifact_locations 默认 []；loop 验证标准 checkpoint |
| activity.artifact.add | activity_id,locations,purpose,source | 追加导航，不把 location 当已验证证据 |
| finding.create（扩展） | claim（原规则） | 原 evidence/状态字段保留；新增 source_activity_id,topics,result_kind,scope,limitations,source_annotation_ids；008 成果提交须满足下述完整性规则 |
| evidence.review | finding_id,evidence_fingerprint,scientific_review,reason,source_activity_id,basis | 不改 EvidenceReference；basis 非空结构化证据数组 |
| relation.create（扩展） | source,target,relation_type | 原 local_id/reason 保留，新增 source_activity_id；008 工作流的修正关系须说明影响范围 |
| annotation.create（扩展） | text | 新增 source_activity_id；原 annotation_target_kind/id 保留；默认 research_state，无 ID；不承担隐式 Finding 语义 |
| agenda.item.create（扩展） | 原必需字段 | 新增 source_activity_id 与 experiment_contract；实验派发时必须完整 |
| dataset.scope.create | project_key,dataset_key,data_revision,description,overlap_keys,coverage,initial_observation,source_activity_id,source | local_id；scope 身份不可变 |
| dataset.observation.record | dataset_scope_id,event_kind,occurred_at,source_activity_id,source,evidence | local_id；candidate_fingerprint 对冻结/评分必需；attempt_id 对评分必需 |
| execution.outcome.record | source_activity_id,agenda_item_id,attempt_key,execution_status,artifact_status,evaluation_status,audit_status,summary,evidence,resources | local_id,retry_of,round,slot_id,native_task_id,requested_settings,actual_settings；attempt_key 在 activity 内唯一 |

新增实体/字段的详细枚举和约束见 [data-model.md](../data-model.md)；输入 status 不接受同义词或任意值。`source` 为 `{kind:string,location:string,note?:string}`，kind/location 非空；它是作者 provenance，不自动验证全部来源内容。evidence 才承担内容身份校验。

## Evidence union

既有 Aim 引用 `{kind:"aim_run",run_hash,role?,note?}` 原样有效。文件引用为新增联合分支；下面展示完整的研究证据形状。

```json
{
  "kind":"local_artifact",
  "location":"results/evaluation.json",
  "sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "size_bytes":123,
  "evidence_type":"evaluation_report",
  "purpose":"估计固定候选在指定窗口的增量及区间",
  "source_activity_id":"$activity"
}
```

```json
{
  "kind":"aim_run",
  "run_hash":"FULL_RUN_HASH_OR_UNAMBIGUOUS_PREFIX",
  "evidence_type":"training_run",
  "purpose":"对照训练指标",
  "source_activity_id":"$activity",
  "role":"baseline"
}
```

kind 必需；role/note 可选。既有 aim_run 继续只要求 run_hash，evidence_type/purpose/source_activity_id 可选，提供时校验；008 工作流须补全这些来源字段。新增 local_artifact 必需 location/sha256/size_bytes/evidence_type/purpose/source_activity_id，resolved_location 是编译器产生的绝对输出字段，不允许客户端写。digest 是内容声明，写入时实际验证，不用示例 digest 伪造身份。只有普通文件；location 相对 repo 根或绝对位置。Aim run 保持规范 full hash resolver。availability/identity_status/scientific_review 是投影字段，不接受作为 create 输入。

如果一次 write 未校验的文件被删除/权限拒绝/digest 不符，整次 update 失败，无 revision。已提交原请求精确重放先走 key/fingerprint，不能因原文件后来变化抹掉历史提交。文件身份无法证明科学结论。

## Finding semantics

既有 finding.create 不新增无条件必填字段。提供 scope 时，其形状为 `{description:string,dataset_scope_ids:entityRef[],conditions:string[]}`，三个子字段必需；提供 topics 时至少一项，limitations 为明确数组，result_kind 四值固定。声明任一结果语义字段（result_kind/scope/limitations）时，必须一并提供这三项、topics、source_activity_id 及至少一条 evidence，保证完整成果而非半结构化主张。008 工作流全部成果提交遵守此完整性规则，旧载荷仍按原规则接受。source_annotation_ids 存在才表示作者确认提炼，原记录可追溯；不得自动把 annotation 里任何路径解析成 evidence。

claim/evidence/scope/result_kind/source 不原地替换。修正创建新 Finding + active relation。Finding epistemic、confidence、governance 可按原操作追加历史；evidence.review 只追加对应 ref 的科学审核历史。

## Experiment / observation validation

experiment_contract 的精确形状见 [autoresearch-protocol.md](autoresearch-protocol.md)。Aimx 校验合同结构、实体引用、scope 的已登记状态；helper/skill 校验实际宿主能力和评分环境。独立确认声明缺完整来源、范围重叠无法判断、已评分/开发或冻结关系不成立，则 independence_claim_rejected（error.details 包含冲突 observation ID/首次事件/完整查询入口）。合法 exploratory 无独立性声明可继续。

observe 事件必须有来源证据，candidate_frozen 要明确 candidate fingerprint；首次评分使用已冻结同一候选与 attempt key，实际 event time 和 commit revision 都保留。晚到报告的 observation 仍追加保存，不因 occurred_at 早于已存事件而丢弃；实际事件时间与 revision 表达不同事实，独立性 gate 按任一评分/开发记录拒绝。新 observation 不使旧 goal criteria 降低，不回滚“已观察”。旧 annotation 不自动成为观察事件。

## 原子性、去重和兼容

先在只读 snapshot 编译，事务内重检 base_revision；所有引用、身份、转换、实验声明均通过才追加 commit/events。事件保留 author、commit/revision；有明确来源时含 source_activity_id，缺来源保持未归属，created_ids 返回原稳定 ID。source_annotation_ids/relations 同 update 可使用已声明引用。

client_update_id 在通用接口中保持可选；008 工作流必需，没有 key 的提交不提供去重/协调保证。request fingerprint 为 canonical ResearchUpdate JSON 的 SHA-256（key 排序、ensure_ascii=False、紧凑分隔）；包含原 base_revision。计算请求指纹前不得注入新增字段默认值或改写原载荷；保留旧请求的规范化算法，确保历史 key 的精确重放仍能取回同一回执。原文件 digest 另算，不能与 request fingerprint 混用。相同 key+相同 fingerprint 返回原 result_json；不同 fingerprint 拒绝。receipt 查不到只说明快照未见，不授权 slot 重放。

SQLite storage schema 和 ResearchUpdate 的 schema_version 均保持 1；旧库现有 idempotency 列仍遵循只在明确写时补齐，读取不 ALTER。回放直接处理已有及新增事件，不维护两套版本路径；缺字段在内存按 unknown/unattributed 显示，不补写历史。未知事件明确报错，不静默跳过。兼容承诺是新程序读取旧数据和接受旧载荷，不承诺旧程序理解未来新增事件。

## 完整可运行 payload 获取方式

[quickstart.md](../quickstart.md) 提供使用临时普通文件、真实 SHA-256 和本期扩展字段的完整 payload 生成步骤。以上例子只说明联合字段，不代表已有命令实现或现成合法数据。
