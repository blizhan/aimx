# AutoResearch 协议：项目能力、实验与交接

本合同更新仓库安装源 `skills/aimx-autoresearch/`，不在规划阶段修改全局已安装 skill。所有研究状态/update/checkpoint 仍由 lead 写；worker 只执行获授权的任务、保留证据并报告，不派生 agent、不重放宿主槽位。

## Config / preflight

现有 config、schema_version=1、规范化算法和 fingerprint 保持原样，继续保存 project_root、aim_repo、goal 原 success_criteria、max_rounds、default_agents、round_overrides、context_budget_bytes。failure_policy（stop_pending 或 continue_independent）与项目能力位置放在本次 preflight/Agenda 实验合同中，由 lead 显式登记并冻结，不塞回已指纹化配置。仍先执行 shape/path validate，再实际 preflight，不把 validate 成功当可实验。

项目 capability 文件：

```json
{
  "schema_version":1,
  "evaluator_protocol":"project-evaluator-v1",
  "preflight_argv":["uv","run","python","tools/evaluator.py","preflight","--json"],
  "evaluation_argv":["uv","run","python","tools/evaluator.py","score","--json"],
  "supports_attempt_log":true,
  "supports_resource_counts":false,
  "enforces_compute_budget":false
}
```

这些是项目提供的入口示例，Aimx 不生成 evaluator 或执行任意描述字符串；使用 argv，不用 shell eval。实际 preflight 只调用已明确授权的项目预检，保存 argv/cwd/退出码/结果身份；不跑训练或修改生产配置/源码。`continue_independent` 的扩展字段 `host_failure_policy_preflight_sha256` 必须逐字匹配 evaluation.preflight_evidence 中一个本地文件引用的 digest，并同时设 `host_failure_policy_verified: true`；孤立布尔声明或未引用报告不能启用继续派发。

预检报告包含以下 checks，每项 status 为 available/missing/unverified，有 verification_source 与 reason：

| check | 必要验证 |
| --- | --- |
| aimx_entrypoint | 实际绝对入口/包版本；复用 version/help/read/dry-run 验证所需命令和证据能力，缺项明确阻塞，无版本协商 |
| research_read | 实际绑定 repo 的 state/overview 可读；与 Aim SDK 能力分开 |
| required_commands | 所用 update dry-run、receipt、evidence/help 接口实际可调用；dry-run 不写 |
| aim_read | 只有合同需要 Aim evidence 才必需；不能因缺 Aim 把文件 evidence/state 误判不可读 |
| models | 所有轮次 requested model/effort 得宿主支持；actual 来源必须原生元数据 |
| evaluator | 项目 preflight 状态、protocol/数据输入 fingerprint/评估产物可读能力 |
| failure_policy | 声明策略与宿主规则一致，continue_independent 有实际宿主支持依据 |
| resource_reporting | 日志/计数/强制能力逐项标识，缺能力可继续声明性研究但不能声称强制预算 |

报告有 schema_version=1、repo/project_root、checked_entrypoint、config_fingerprint、failure_policy、checks、ready、limitations。required 缺失或 unverified 则 ready=false，不派发实验。没有可验证失败策略时停止后续派发，报告实际缺项；不以“协议待升级”代替具体检查。旧循环能按原样读取，继续派发前补齐关联合同/预检即可，无格式转换步骤。

## Frozen experiment contract

```json
{
  "contract_version":1,
  "question":"固定候选是否在此前未观察窗口改善指标？",
  "interventions":["仅改变候选规则"],
  "fixed_controls":["相同输入、样本范围、基线和 evaluator"],
  "expected_evidence":["完整预测和配对评估报告"],
  "disconfirmation":["项目定义的独立确认阈值未满足"],
  "confounders":[],
  "evaluation":{
    "protocol_id":"project-evaluator-v1",
    "protocol_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "argv":["uv","run","python","tools/evaluator.py","score","--json"],
    "metric":"project_metric",
    "acceptance_rule":"项目预先给出的阈值与区间规则",
    "preflight_evidence":[]
  },
  "data":{
    "scope_ids":["ds_example"],
    "phase":"independent_confirmation",
    "candidate_fingerprint":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "observation_revision":4
  },
  "resources":[{
    "kind":"evaluator_calls",
    "unit":"calls",
    "planned":1,
    "enforcement":"declared_only",
    "verification":"unverified"
  }],
  "failure_policy":"stop_pending"
}
```

示例 fingerprint/IDs 必须换真实值；完整合同 evaluation.preflight_evidence 至少一条可验证 local_artifact ref，扩展字段按 object 原样保存。question 非空，其余数组必需；interventions/fixed_controls/expected_evidence/disconfirmation 至少 1 项，confounders 可以为空但联合干预缺少混杂解释即不完整。metric/acceptance_rule/协议身份/候选指纹/数据范围/资源上限必需；planned 为非负数且有限，不允许 NaN/Infinity。资源标记 verified 时必须同时提供 actual 数值和 executor source；否则标记 unverified。promoted_to_development 也必须指明此前冻结的相同 candidate fingerprint。

phase 为 exploratory 或 independent_confirmation；本期不通过“validation/holdout”名称自动判独立。declared_unobserved + declared_complete 来源、candidate_frozen 事件和全部范围无评分/开发冲突才可用后者；未知观察状态拒绝该声明。全部选中范围及其他已登记范围必须有可证明的非重叠 overlap keys；未知 keys 或任意交叠都拒绝独立确认。已评分范围仍能做 exploratory，不减少原 goal 标准。首次评分必须记录对应 frozen candidate/attempt；之后不能重复称未观察。

开始 slot 前：核对 preflight→保存合同与来源→核对 observation revision→begin-round 消费轮次→reserved 持久化→spawn→started/native_task_id。若在最后校验后有并发观察事件或在评分启动前 scope 变更，项目 scorer 必须重新核对并拒绝独立性声明；Aimx 的单次快照不提供评分 lease。首次评分/转开发必须先登记同一 ResearchUpdate 中较早事件的 attempt/fingerprint，提交顺序表达逻辑先后；真实 occurred_at 与创建 revision 始终保留。迟到报告的事实作为 append-only observation 保存，不以入库先后覆盖实际事件时间，但从该快照起独立确认声明会被拒绝。不能声称记录系统证明无未记录标签读取。

## Attempt result

Attempt 报告保留 summary/evidence/candidate_ref 与 requested/actual metadata，增量补充四个事实轴；现有宿主返回 status 时原样保留，缺新轴显示 unknown，不推断科学成功：

```json
{
  "schema_version":1,
  "attempt_key":"loop-1-r1-slot-a-native-1",
  "execution_status":"failed",
  "artifact_status":"complete",
  "evaluation_status":"passed",
  "audit_status":"not_run",
  "summary":"服务尾部失败，完整预测已保存且项目评分已完成",
  "evidence":[],
  "resources":[{
    "kind":"model_calls","unit":"calls","planned":1,"actual":2,
    "verification":"verified","enforcement":"declared_only",
    "source":"executor attempt log"
  }]
}
```

科学评价通过必须有真实评估证据，例子需补真实 ref 后才合法；无法查证字段使用 unknown，不从 task completion 推断。worker 原始结果不可覆盖，lead 核验后提交 execution.outcome.record/Finding/关系；无科学主张可只记录 annotation/outcome。负结果不改 execution_status；整体 goal 独立按原 criterion 评估。

执行器报告多个 attempt 时各有不同 key 和 retry_of，日志缺失/重复/超额出现在回执；统计实际执行，不给失败 attempt 退款。多 attempt 记录不允许助手重新 spawn reserved/started/finished slot。缺 actual_settings 就省略并显示未核验，不复制 requested 充当 actual。

## Failure policy / checkpoint compatibility

- stop_pending：已知失败或基础设施错误后不派发后续 pending slot；已在运行的任务按宿主规则保留，不承诺取消进程。
- continue_independent：仅在派发前预检宿主明确允许时继续真正独立任务；基础设施失败仍遵守宿主更严格规则。
- 无可验证策略：停止后续派发，checkpoint blocked/原因，输出回执。
- 模型、effort、轮数、原标准固定；不自动扩轮、替换模型、轮内重放或退款。

control.py 的 validate/init/begin-round/checkpoint/status 入口和 checkpoint 结构继续使用现有规则，无 upgrade 命令、备份格式或第二套状态机。preflight 由 skill 组织实际检查并形成报告；报告、实验合同、Attempt 四轴与 update digest 显式写入关联研究记录/产物，通过既有 artifacts、update_file、research_update 回执关联。checkpoint 中 finished.result 保留现有 status/summary/evidence 等形状，不把新 outcome 字段硬塞进去，也不改原 status；overview 将它与独立 outcome 事实并列呈现差异。

继续旧循环前先协调原提交、检查新合同/策略与观察记录；缺什么就阻塞什么，不修改已保存 config/fingerprint、已完成 slot 或终态。新循环也使用同一 checkpoint 结构和步骤。终态补证使用独立活动。

## Commit / receipt / finish

保存 exact update 文件及 digest→dry-run→commit→checkpoint key/commit_id/revision。写库成功但 checkpoint 中断，先 receipt key/fingerprint 查询或原请求精确重放，不能依据缺回执再次生成内容。指纹冲突停止协调；revision conflict 刷新上下文、人类 steering 和观察状态后重审，用新 key。

无论结束、暂停、阻塞、失败，lead 都调用 overview/Receipt，输出原 goal 判断、预算、loop 与长期方向、成果（含负结果）、实际 repo/state/checkpoint/report、持久化身份、待协调事项、可复制查询。状态库不可读时保留产物导航和明确部分失败，不称知识已入库。

## Follow-up

终态后创建独立 follow_up Activity，新增文件证据 Finding，通过 supports/challenges/refines 关联原 Finding。不调用原 begin-round/checkpoint，不改原 budget/assessment/status。原 loop 查询将 follow_up 单列，主题 show/context 可以带关系展示双方范围。历史扩展与未来独立确认必须按真实数据观察阶段命名。
