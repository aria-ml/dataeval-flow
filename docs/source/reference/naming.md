# Naming Conventions

Every name a user reads or writes follows one rule per kind of name. The rules below are for plugin authors and for
anyone porting new functionality. `tests/test_naming_conventions.py` holds the built-in steps to them, and a failure
there names the rule. Plugins are asked to follow the same rules. The registry enforces none of them, though the
`drift-monitoring` and `ood-detection` presets refuse their reserved detector names at load.

Each rule gives the guard test that enforces it on built-ins. A rule marked "convention, no guard test" is followed by
every built-in step but checked by no test.

The steps named here are in the [evaluator](evaluators.md), [transform](transforms.md), [combine](combines.md) and
[check](checks.md) catalogs, and the presets are in the [preset catalog](presets.md).
[Find the Right Step](index.md) leads from a question to the steps that answer it.

## Step types and titles

### The type matches the title

A step's `type` and its `title` agree once case, spaces, hyphens and parentheses are dropped. Both are lowercased and
stripped of everything except letters and digits, and the results are equal. This holds for every kind, presets
included. `drift-mmd` is titled Drift (MMD), `outliers-by-class` is titled Outliers by Class, and `kfold` is titled
K-Fold.

A check's findings carry its title, so the name a user reads in a report is the key written in `checks:`.

Guard test: `test_a_step_type_and_its_title_name_one_thing`.

### Types are unique across kinds

No two registered step types share a name, whatever their kinds. `outliers` is an evaluator, and the combine that
groups it by class is `outliers-by-class`, so each name picks out one step.

Guard test: `test_no_two_step_types_share_a_name_across_kinds`.

### Each kind has a pattern

Convention, no guard test. No test reads a step type for a suffix, so the check rule below is checked by eye.

- **An evaluator** is a noun for what it measures. Where it wraps one DataEval class, it takes that class's name:
  `balance`, `coverage`, `duplicates`. Detector variants are `<family>-<method>`, titled `Family (Method)`:
  `drift-kneighbors`, `ood-domain-classifier`. Flow's own evaluators take a subject prefix:
  - `label-` for labels and classes (`label-health`, `label-alignment`);
  - `factor-` for one metadata column (`factor-summary`, `factor-triage`);
  - `ontology-` for the ontology (`ontology-validation`);
  - `content-` for item bytes (`content-digest`).
- **A combine** is named for what it makes: `outliers-by-class`, `ood-union`, `factor-gaps`.
- **A check** is named for what it judges, never for the statistic it uses, so it carries no `-rate` or `-score` suffix:
  `image-outliers`, `uncovered-items`, `dimensional-completeness`.
- **A transform** is a verb: `split`, `select`, `remove`, `merge`.
- **A preset** is named for the question it answers: `data-cleaning`, `drift-monitoring`.

### A description is one sentence

A step's description starts with a capital letter and ends with a period. It holds one sentence, on one line, and
never a second one. The registry shows it as the step's entry in the catalogs.

Guard test: `test_a_step_description_is_one_sentence`.

### A check's findings carry its title

A check's findings are titled with the check's title, so `image-outliers` makes a finding titled Image Outliers, and
`class-imbalance` makes one titled Class Imbalance. This is a convention with no guard test.

Two checks title their findings differently, on purpose:

- **`metadata-issues`** makes one finding per kind of issue it finds, plus Suggested policy, Verified and Recommended
  policy. Its titles carry the structure of the metadata report. Its key is still `metadata-issues`.
- **`drift` and `ood`** title each finding with the detector's subject: the `subject:` setting, or, where none is
  set, the evaluator's title followed by its entry's name where that differs from its type, as in Drift (MMD) · mmd.
  A chain with several detectors can then tell their findings apart.

`ood-agreement` makes two findings that share its title, OOD Agreement. Its description tells them apart: the share
every detector flagged, and the images one detector alone flagged.

Three names are kept on purpose:

- `metadata-issues`, the check on `factor-triage`'s output, keeps its name, though its evaluator takes the `factor-`
  prefix.
- `kfold`, the one transform that is not a verb. Folding it into `split` would make a port's list-ness depend on
  config, which is engine code.
- `eval-coverage`, whose short "eval" matches the `evals` ports of `class-sufficiency` and `untrained-classes`.

## Ports and settings

All three rules here are conventions with no guard test.

- **Ports.** `input` is a step's main input. Any other input is a noun for what it carries, such as `reference`,
  `ranking` or `alignment`, and is plural when it takes a list: `evals`, `duplicates` and `factors`. Some ports that
  take no list are plural too, for what they carry: `labels`, `outliers`, `plans` and `parts`. Outputs are `output`,
  or named outputs such as `train`, `val` and `test`. A check's output is `findings`.
- **Evaluator settings** use DataEval's parameter names verbatim. Flow's own settings are plain snake_case words:
  `chunking`, `per_image`, `per_target`, `factors`, `verify`. A transform that calls a DataEval function keeps that
  function's names (`test_frac`, `val_frac`, `stratify`, `split_on`), and `kfold`'s `folds` is Flow's own word.
- **Policy references** are `metadata:` and `stats:` on every step that reads them.

## Check thresholds

A check's settings follow these rules. `test_a_check_names_its_bounds_by_what_they_bound` enforces three of them: a
single bound is called `warning`, `info` comes with `warning`, and no setting is named for a statistic. The other rules
are conventions with no guard test; `steps/checks/_limits.py`'s `exceeds` implements the boundary rule.

- **One quantity.** A check that bounds one quantity calls its bound `warning`, and its second bound `info` where it
  offers one. `info` comes with `warning`. `image-outliers` has `warning`, and `distribution-shift` has `warning`
  and `info`.
- **Several quantities.** A check that bounds several quantities names each bound for its quantity, and each bound
  warns: `image-duplicates` has `exact` and `near`, and `class-sufficiency` has `train` and `eval`.
- **No statistic names.** A bound is named for what it bounds, never as a `rate`, `count`, `ratio` or `total`, or with a
  `_rate` suffix.
- **Plain-word switches.** A switch is a plain word: `empty`, `declared`, `warn_on_drift`.
- **Display settings** keep their own names and bound nothing: `max_examples`.
- **The boundary is strict.** A bound is the last value that does not warn. A finding warns only when the measured
  value passes the bound: strictly greater than an upper bound, strictly less than a lower bound. `info` works the
  same way for the `info` band.
- **`null` turns a bound off.** A check with no bound to judge by reports its finding as information.

## Preset settings

A preset's settings follow these rules. Each rule names its guard tests, or is marked as a convention.

- **Spelling.** A preset spells a setting as the step that takes it spells it, with the same meaning, and with no step
  prefix. The preset may require a setting the step leaves optional, narrow its values, or give it its own default. It
  never widens or retypes it. Two parts are tested: `test_a_preset_spells_no_setting_with_a_step_prefix` (no
  `outlier_` or `duplicate_` prefix) and `test_a_presets_step_block_holds_only_that_steps_own_settings` (a block holds
  only settings its step takes). The meaning and the narrowing rules are conventions with no guard test.
- **The main step's settings sit at the top level.** A preset built around one main step keeps that step's settings at
  the top, though its chain runs more steps: `data-splitting` takes its `split` or `kfold` step's `test_frac`,
  `val_frac` and `folds`, and `metadata-triage` takes its `factor-triage` step's `verify` and `default_bins`. The
  `detectors:` lists of `drift-monitoring` and `ood-detection` sit there too. Convention, no guard test.
- **Other steps sit under their type.** Every other step's settings sit under a key named for its type. A block may take
  a subset of what that type's own entry would hold, leaving out what the preset fixes or does not offer, but never a
  setting the step does not take. `data-coverage`'s `coverage:` block holds `coverage`'s settings, and `data-cleaning`'s
  `duplicates:` block holds five of `duplicates`' settings. Setting a block to `false` switches its step off where the
  preset already allows that, as `factor-gaps: false` does on `data-coverage`. Its `completeness: false` switches the
  completeness steps off too, though `completeness` is a switch, `true` or `false`, rather than a block. Guard test that
  a block holds no setting its step does not take:
  `test_a_presets_step_block_holds_only_that_steps_own_settings`. The `false` switch is a convention with no guard test.
- **`checks:` is keyed by check type.** Each value holds that check's own settings, spelled in the check's words, and
  never a setting the check does not take. `data-cleaning`'s `checks:` has an `image-outliers` key holding
  `warning`. Guard test: `test_a_presets_checks_are_keyed_by_check_type_in_each_checks_own_words`.
- **Preset-wide settings stay flat.** `metadata`, `stats` and `ontology` apply to the whole chain. So do choices that
  no single step takes verbatim: `data-splitting`'s `rebalance` and `drift-monitoring`'s `classwise`. Convention, no
  guard test.

The [preset catalog](presets.md) lists each preset's settings and `checks:` defaults.

## Step names in a preset chain

These names reach users as `Finding.step` (`class-imbalance[train]`), as the keys of a chain's results and as evaluator
banners, so they follow one rule. Guard test for the first two rules:
`test_a_presets_steps_are_named_for_their_types`.

- **Named for its type.** An evaluator, combine or check step is named for its type, and a preset's own evaluator
  entries are named for their type. `data-cleaning` runs `outliers`, then `image-outliers`.
- **`<type>-<role>` for a second role.** When one type serves several roles in a chain, the role is appended:
  `data-prioritization` runs `outliers-reference` and `outliers-pool`. The test accepts
  `<type>` or `<type>-<anything>`, so it does not check the role's name.
- **A transform is named for the Dataset it makes.** `data-cleaning`'s `remove` step is named `clean`, and
  `data-prioritization`'s `select` step is named `selected`. The test skips transforms, so
  this rule is a convention with no guard test.
- **A detector entry keeps its name.** A step over a user-named detector entry takes the entry's name, and the steps
  made per entry add a suffix: `<entry>-check`, `<entry>-by-class` and `<entry>-by-class-check`. Two presets refuse
  reserved detector names when the config loads:
  - `drift-monitoring` refuses a name ending in `-check`, `-by-class` or `-unchunked`.
  - `ood-detection` refuses `ood-union`, `ood-agreement`, `factor-predictors` and `factor-deviation`, and any name
    ending in `-check`. It has no `-by-class` steps.

  The naming test skips detector steps, so the step-name pattern itself has no guard test.

## Registering a step

A plugin registers a step through an entry point in its package's metadata, in its kind's group:
`dataeval_flow.evaluators`, `dataeval_flow.transforms`, `dataeval_flow.combines`, `dataeval_flow.checks` or
`dataeval_flow.workflows`. The entry point's name is the step's type, and its value names the class as
`module:attribute`. The class subclasses its kind's base, {py:class}`~dataeval_flow.evaluators.Evaluator`,
{py:class}`~dataeval_flow.steps.Transform`, {py:class}`~dataeval_flow.steps.Combine`,
{py:class}`~dataeval_flow.steps.Check` or {py:class}`~dataeval_flow.workflows.Workflow`, and its `name` is the entry
point's name. An evaluator's or workflow's config also configures that type and declares its `inputs`. A plugin that
fails these is logged and left out, and a plugin can never take a built-in's name.

## Python names

| Name | Form | Example | Guard test |
| --- | --- | --- | --- |
| Step class | `<Type><Kind>` | `ImageOutliersCheck`, `OutliersByClassCombine` | `test_a_step_class_is_named_for_its_type_and_kind` |
| Step config | `<Type>Config` | `ImageOutliersConfig`, `DriftConfig` | `test_a_step_config_is_named_for_its_type` |
| Flow's own Output | `<Type>Output` | `FactorSummaryOutput`, `OutliersByClassOutput` | `test_an_output_flow_defines_is_named_for_its_type` |
| A preset's `checks:` model | `<Preset>Checks` | `DataCleaningChecks`, `MetadataTriageChecks` | convention, no guard test |
| A block keyed by a step type | `<Type>Settings` | `OutliersSettings` | convention, no guard test |
| The same, where presets differ | `<Preset><Type>Settings` | `DataSplittingClassImbalanceSettings` | convention, no guard test |

- **Step classes** are `<Type><Kind>`, with the kind as the suffix. A class's name lowercased equals its type squashed
  plus its kind.
- **Step configs** are `<Type>Config`. Two are named `<Type><Kind>Config`, because `<Type>Config` already names a
  pipeline pool entry: `ViewTransformConfig` (`ViewConfig` is the `views:` entry) and `ExportTransformConfig`
  (`ExportConfig` is the `exports:` entry). The guard test names both exceptions.
- **Outputs.** Flow's own Output is `<Type>Output`. The guard test checks only an Output that one type makes. Two
  rules have no guard test: an evaluator whose Output is DataEval's class keeps DataEval's name, such as
  `BalanceOutput`, and evaluators that share one Output name it for their family.
- **Preset settings models** are named for the key they model. A preset's `checks:` map is `<Preset>Checks`. A block
  keyed by a step type is `<Type>Settings`. Where more than one preset defines its own model for one type, each is
  `<Preset><Type>Settings`, because the config schema keys its `$defs` by class name and two models with one name
  would get mangled names. Where two presets take the same block they share one model: `data-prioritization`'s
  `cleaning:` reuses `data-cleaning`'s `OutliersSettings` and `DuplicatesSettings`.
- **One home per public name.** A name is exported from one package. `Finding` is exported from `dataeval_flow.steps`,
  beside `Check`, `CheckConfig` and `StepResult`, and not from `dataeval_flow.workflows`. Guard test:
  `test_finding_is_exported_beside_check_and_only_there`. The `list_<kind>s` and `get_<kind>` helpers stay with their
  kind's package, a convention with no guard test.
