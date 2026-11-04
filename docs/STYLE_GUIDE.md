# Documentation style guide

This guide records how DataEval Flow documentation is written. Follow it when you add or change a tutorial, a how-to
guide, an explanation, a reference page, a report or a docstring. It follows the DataEval
[documentation style guide](https://github.com/aria-ml/dataeval/blob/main/docs/STYLE_GUIDE.md) wherever the two projects
work the same way, and it says where Flow differs.

Each rule says how it is checked. A rule that names a tool or a test fails the build when you break it. A rule marked
"review" has no automated check, so a reviewer holds you to it. Flow checks far less automatically than DataEval does:
no test reads the prose or the structure of a tutorial. The last sections list the commands to run before you open a
merge request.

Pages written before this guide may not follow it yet. Bring a page into line when you edit its prose.

## Choose the kind of page

Flow follows the [Diataxis](https://diataxis.fr) framework. Every page has one purpose.

| Kind | Purpose | Lives in | Name |
| ---- | ------- | -------- | ---- |
| Tutorial | Teaches by working end to end on a realistic dataset, from the data to reading the report. The reader learns by doing. | `docs/source/notebooks` | `<task>.py`, in snake case |
| How-to guide | Solves one task for a reader who already knows what they want. A recipe. | `docs/source/how_to` | `<task>.md`, in snake case |
| Explanation | Explains an idea, the reasoning behind it and its limits. | `docs/source/concepts` | `CamelCase.md` |
| Reference | Lists every step, preset, setting and term: the catalogs, the glossary, the maturity and naming pages. | `docs/source/reference` | lowercase `.md` |
| Home | Orients a new reader: installation, the quickstart, the changelog and the release highlights. | `docs/source/home` | lowercase `.md` |
| Generated reference | The API reference comes from docstrings, and the config reference from `config/params.schema.json`. | `src/dataeval_flow` | not applicable |

A tutorial is heavy and a how-to is light. A tutorial takes the reader through a whole task and ends by summarizing what
they learned. A how-to is the shortest path to one result.

Rules (review):

- Teach through links. A how-to does not explain why something works. It links to the tutorial that teaches the topic
  and the explanation that covers it. A tutorial links to an explanation for depth and does not repeat it.
- Write a walk-through of a whole task as a tutorial, even when it covers one evaluator and one check. Everything else
  is a how-to. A how-to may be a notebook, as `onnx_embeddings.py` is, when the recipe needs executed output. List it in
  `how_to/index.md`.
- Give a how-to one task. If it needs two recipes, write two how-tos.
- Link to the page that owns a recipe instead of teaching it again.
- Call a page by its kind. A how-to says "this guide" and not "this tutorial".
- Name a step by its type, in backticks, and link its first mention to its catalog entry:
  ``[`sufficiency`](../reference/evaluators.md#sufficiency)``. Do not write "the sufficiency tool".
  `tests/test_catalog_docs.py` fails when the anchor of such a link in a catalog page differs from the name in its text.

## Write the project files

The Markdown files at the root of the repository, and `docs/CACHE_MANAGEMENT.md`, are not Diataxis pages. They follow
the prose rules and the heading rules in this guide. CI runs markdownlint over them and checks their links.

- Write `README.md` as a home page. It is the front page on GitHub and PyPI, so keep its links absolute.
- `CONTRIBUTING.md`, `BRANCHING.md`, `SECURITY.md`, `ROADMAP.md`, `AGENT.md` and `docs/CACHE_MANAGEMENT.md` state a
  process, a policy or a plan. A policy keeps the strength of each "must", "shall" and "should".
- Leave `CODE_OF_CONDUCT.md` as the published Contributor Covenant text. Leave `CHANGELOG.md` in the format that
  `BRANCHING.md` sets, since `scripts/release.py` parses it.
- Do not edit `ironbank/*/README.md` by hand. `ironbank/generate.py` renders it from `ironbank/README.md.j2`, so change
  the template and run `nox -s ironbank_gen`.

## Write the prose

Narrative text is written to the reader in the second person and kept plain. This applies to tutorials, how-to guides,
explanations and the home pages. It does not change the docstring rules in the docstrings section.

### Voice

- Use the directive voice: "You should check ...", "You will create ...", "You can drop ...". Do not write "Note that
  one can ...", and do not describe what the library does when the sentence tells the reader to do something.
- Write short declarative sentences. Two plain sentences are better than one sentence with a colon, a dash and a
  dependent clause.
- Write contractions out: "do not", "it is", "you will". A possessive `'s` is not a contraction and stays.
- Use American spelling. `codespell` runs with its default dictionary and skips the notebooks, so it does not convert
  British spellings. Review checks the spelling.
- Say "dataset" and never "corpus". Flow is a computer-vision tool, and "corpus" reads as text terminology.
- Write for a reader with no statistics background where a page teaches how to read a result. Translate a technical
  statement and keep its qualifier: "at least 95.6% right, and you can be 95% sure". Keep the precise term in the
  report, the reference and the docstring, and add a short glossary to the tutorial.

### What to avoid

| Avoid | Write instead |
| ----- | ------------- |
| A metaphor or figurative framing: "retire the sentinels", "a factor sat out the analysis" | The plain verb: "replace the sentinels", "one factor is still dropped" |
| An em-dash aside or a trailing flourish | Two plain sentences, or cut it |
| Rhetorical stress: italics for emphasis, bold stress, "exactly", "simply", "genuinely", "quietly", "on your behalf" | Delete the word or the markup |
| A not-X-but-Y contrast, a one-line closer, a forced triad, filler or sales language | Say the point once. Keep a contrast that carries a real qualification |
| A figurative heading, such as a numbered "lens" | A heading that names what the section does |
| Commented-out code or links left in a notebook | Remove them. Keep a commented alternative only when the text explains it |

The only em dash allowed is the separator in a Next steps entry (see the Next steps section).

No test reads Flow's prose. Review checks all of it, in every kind of page.

### Emphasis

Do not use italics for emphasis. Use bold only at the first mention of a term that the page defines. Do not use bold
for a list-item label, a heading or stress. Use `*` for emphasis markup and never `_`, because a dunder such as
`__init__` is itself valid strong emphasis.

These rules cover the cells of a table too. Do not bold a row label, and do not put an em dash in a cell. Write a column
heading in sentence case, with a capital on the first word only.

One exception covers the reference catalogs. The `- **Label:**` bullets of a catalog entry, such as `Combined by` and
`Used in`, are the entry template, and `tests/test_catalog_docs.py` reads them by their bold label. Keep them bold. A
catalog entry's heading is the step's type name and is not cased as prose.

markdownlint (MD049 and MD050) flags a Markdown page that mixes `*` and `_`. It does not read the Markdown cells of a
notebook. Review checks the rest.

### Claims, numbers and limits

- State every limit, assumption and caveat the reader needs, in plain words. Do not soften or remove a hedge when you
  edit for style.
- Say what a result rests on. A score, a bound or a projection comes with the data it was measured on and the
  assumptions it needs, in the same sentence or the next one.
- Quote a number in prose only after you have checked it against the executed output. Protect it with an assertion cell
  (see the code cells section).
- Do not state a cause you did not trace. Write what you measured.
- Do not describe a library behavior in a docstring or page without running it.
- Do not describe future work as if it were a feature ("to follow", "planned").
- Follow the advice you give. A tutorial that says to divide the error budget across runs divides it in its own runs.

### Line length and Markdown

- Wrap Markdown at 120 columns. markdownlint enforces this in `.md` pages. Tables and code blocks are exempt
  (`.markdownlint.yaml`).
- Wrap the Markdown cells of a notebook at 120 columns, counting the `#` and the space that start each line. `ruff`
  ignores line length in `docs`, and markdownlint cannot read a `.py` file, so review checks it.
- Run `uv run nox -s lint` before you push. It runs `ruff`, `ruff format` and `codespell`.

## Write reports

A report is the text or HTML page a finding produces. It is read at a glance, so it follows rules of its own. They apply
to the blocks a check, an evaluator or a workflow builds, and to the text that goes in them.

Rules (review):

- Be succinct. Draw a finding as a table with a bar, stacked or sparkline column, add at most one lede sentence, and
  leave the detail in `result.json`. Use ASCII drawing where it says more than a sentence.
- State only what this run found. A fact that is true of every run belongs in the reference and the tutorial, and not in
  each report: a check's standing assumption, what a mark means beyond a one-line key, the next steps, a pointer to
  `result.json`.
- Put a mark in a column of its own, headed `☐`, and explain the marks in one key line under the table.
- Build a report from the existing block types in `src/dataeval_flow/_blocks`. A new block type changes the contract
  with the studio that reads `results.json`, so it is a decision for the maintainers.
- Read the whole of `result.report()` when you measure a change, including the section of the evaluator step that
  feeds the check, so that nothing is said twice.
- Title and describe a finding as [Naming conventions](source/reference/naming.md) says.

## Structure a tutorial

A tutorial's sections come in this order. The headings are fixed strings.

| Section | Level | Required | Content |
| ------- | ----- | -------- | ------- |
| Title | H1 | yes | An imperative verb phrase, for example `Clean a dataset` |
| Opening | | yes | One or two sentences that name the step or workflow and the dataset |
| Audience and role | | yes | No heading. Two short paragraphs, with no bold label: who the tutorial is for, and where the task falls in a larger workflow, with a link to the tutorial before it and the one after it |
| `What you will do` | H2 | yes | The steps, as a list |
| `What you will learn` | H2 | yes | The outcomes, as a list |
| `Prerequisites` | H2 | yes | The packages, the data and the network access the notebook needs |
| `Lay out the data root` | H2 | yes | See below |
| Working sections | H2 | yes | One per step, named for what the reader does |
| `Conclusion` | H2 | yes | "In this tutorial, you learned how to:" and a list |
| `Next steps` | H2 | yes | See the Next steps section |
| `On your own` | H2 | yes | How to repeat the analysis with the reader's own data, model or settings, one entry for each setting to change |

Lay out the data root right after the prerequisites. Flow reads and writes its files relative to a data root, which is
the folder the notebook runs in. Show what that folder holds once the tutorial has run, as a tree in a `text` fence with
one comment for each entry. The tree is true of that notebook: it lists the data, the models, the trainers and the
output the notebook reads or writes, and says in a sentence what it does not read. It also says where the pipeline
lives, which is a YAML file next to the notebook or the Python the notebook builds.

A tutorial that runs a task shows both reports. Print `result.report()` for the text report. For the HTML report, call
`show(result.to_html())` in a cell tagged `remove_input`, with the `show` helper that
[View a report as HTML](source/notebooks/view_html_reports.py) defines.

Review checks the order of the sections and the data-root tree.

## Structure a how-to

A how-to is a Markdown page.

| Section | Level | Required | Content |
| ------- | ----- | -------- | ------- |
| Title | H1 | yes | An imperative verb phrase, for example `Reuse a cleaning chain on new data` |
| Lead | | yes | No heading. The problem and the step that solves it, with a link to its catalog entry and its explanation, then what the reader needs beyond a working pipeline |
| Working sections | H2 | yes | One per step, named for the action |
| `Next steps` | H2 | yes | See the Next steps section |

Keep a how-to short (review):

- Stay on the task. Leave out topics the task does not need. Link the tutorial or explanation that owns the reason.
- Write a step as a short lead-in, the config or code, and at most two sentences on what to read in the output.
- Cut a sentence that repeats what a code line says or what an output just showed. Keep every limit, assumption and
  caveat.
- A how-to that carries two tasks or several alternative paths is two how-tos. A how-to that has to teach a concept to
  be useful is a tutorial.
- Do not close with `Introduction`, `Summary`, `Conclusion`, `See also` or `Related material`. Put the links the reader
  needs in the lead, in the step that uses them, or in `Next steps`.

## Structure an explanation

An explanation is a Markdown page in `docs/source/concepts`. It explains one idea, the reasoning behind it and its
limits, and it does not walk through a task. End it with these sections, in this order, when they apply:

| Section | Level | Content |
| ------- | ----- | ------- |
| `Related concept pages` | H2 | The explanations a reader needs next, each a link and a one-line description |
| `See this in practice` | H2 | Lists headed `Tutorials`, `How-to guides` and `Reference` at H3, each entry a link and a one-line description |
| `References` | H2 | The papers and sources the page cites, numbered |

Write each entry in these lists as a link, a colon and a one-line description, such as
`[Check a set of splits](../how_to/check_a_set_of_splits.md): Run an audit step on the parts of a split.` The em dash
belongs to `Next steps` and `Where to go next` entries only. Link a tutorial or how-to guide by its exact title.

Review checks the order and the endings. Do not close an explanation with `Summary` or `See also`.

## Structure a home page

Home pages orient a new reader: the installation, the quickstart, the changelog and the release highlights.

- Name each section of the installation and the quickstart for the action it covers, such as `Install DataEval Flow` or
  `Describe a pipeline`. Position carries the order, so write no `Step N:` and no numbering.
- End the quickstart with `Where to go next`, in place of `Next steps`. Write each entry in the form of a Next steps
  entry: a link, an em dash and one sentence.
- The release notes in `home/whats_new` are a record. Bring the formatting of an older page into line with this guide
  and leave what it says as it is.

## Write headings

These rules apply to every heading in a notebook or a page. Apply them to the page titles, the sidebar titles and
captions of a toctree, and the titles of cards and list-table rows as well.

| Rule | Statement | Example |
| ---- | --------- | ------- |
| Sentence case | Capitalize the first word and proper names only | `Marginal class probability shift`, not `Marginal Class Probability Shift` |
| Titles | Tutorials and how-tos use an imperative verb phrase. Omit the `How to` prefix. No `Introduction to`, no gerund, no colon subtitle | `Clean a dataset`, not `Introduction to data cleaning` |
| No numbering | No `Step 3:` and no `3.`. Position carries the order. A number goes stale when steps move, and it is part of the anchor that links use | `Run the task`, not `Step 3: Run it` |
| No colon subtitle | Name the section for what it does, without a category before it | `Load the dataset`, not `Data Preparation: Load the dataset` |
| Plain text | No bold, no em dash and no trailing colon. Use backticks only around an identifier | `Quick decision table`, not `**Quick decision table:**` |
| One H1, no skips | A page has one H1, and heading levels never skip | |
| Fixed template headings | Use the strings in the two structure tables | `What you will do`, not `What you'll do` |

When you refer to a section in prose, name it: `see the "Run the task" section`. Do not refer to it by number.

Review checks the headings. A heading anchor is built from its text, so a heading that carries a number changes its link
when the number changes.

## Write the Next steps section

Every tutorial and how-to ends with `Next steps`. Each entry has this form, and the em dash is the only one allowed in
documentation prose. In a notebook, use the `{doc}` role with the target's file name. With no link text, the role
inserts the title of the target page. In a Markdown page, use a Markdown link.

```text
- {doc}`data_prioritization` — Rank an unlabeled pool so the most informative samples are annotated first.
- [Reuse a cleaning chain on new data](reuse_a_workflow.md) — Keep a cleaning chain in your config and run it on new data.
```

Rules (review):

- List two to four entries in a tutorial and two or three in a how-to.
- Include at most two explanations and at most two tutorials.
- Choose the explanation of the step or topic the page is about. Add a second only when the page relies on a second
  distinct idea. Leave out general pages unless they are the topic.
- For a how-to, choose the tutorial that teaches the topic first, then the natural next task. For a tutorial, choose
  the how-to that applies its main step in depth, or the next tutorial in the learning path.
- Leave out a link the body already makes for a side detail, such as an optional setting. Keep the topic's explanation
  and the natural next task even when the body links them. Link the reference entry of a step in the body, where the
  step is first named, and not in `Next steps`.
- Use the exact title of the target page as the link text, and a one-sentence description. Do not repeat the link text
  in the description.
- Put a suggestion that is not a link, such as "use your own model" or "average more runs", in `On your own`.

## Write the code cells

### Format

- Write the notebook as a `py:percent` script in `docs/source/notebooks`. The `.py` file is the committed source. The
  `.ipynb` file is generated and ignored by git.
- Copy the header from a neighboring notebook, including its `jupytext_version`.
- `uv run nox -s docs` converts the scripts for you. `uv run nox -s docsync` also copies a newer `.ipynb` back into its
  `.py`, so run it only when you edited the notebook in Jupyter and mean to keep those edits.
- Import from the public modules listed in [Contributing](../CONTRIBUTING.md#where-code-lives). Never import from a
  module whose name starts with an underscore.
- Keep notebooks self-contained. Download a dataset in the first cells with `maite-datasets`, write models and outputs
  under the data root, and list what the reader needs under `Prerequisites`. Add a dataset to `docs/datasets.toml`,
  which the docs dataset cache restores before a build and `uv run nox -s docs_data` downloads.
- Comment only what is not obvious from the code, and keep the comment short.

### Hide cells with tags

Tag a cell whose code or output is not the point of the page. The cell still runs, and the `.py` file shows it in full.
Do not hide a cell that is part of the recipe, and do not hide a cell whose code the text refers to.

| Tag | What the page shows |
| --- | ------------------- |
| `remove_output` | The code, without its output. Use it for a download, an install or a progress bar |
| `remove_input` | The output, without the code. Use it for the HTML report |
| `remove_cell` | Nothing. Use it for an assertion cell and a helper |
| `hide-output` | The code, with its output collapsed |

### Assertion cells

A sentence that states a result needs a check that fails when the result changes. Put the check in a hidden cell:

```python
# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert result.success
assert 0 < len(outliers) <= 0.05 * len(dataset)
```

The docs build executes a notebook whose code cells changed and fails if an assertion fails
(`nb_execution_raise_on_error`). Assert a threshold or a range for anything that varies between hardware or runs.
Training results differ between CPU and GPU, and some statistics vary slightly between runs. When you merge, trim or
delete a notebook, keep every assertion cell it had or replace it with an equivalent one.

### Seeds, devices and output

- Set the seed once, in the pipeline's `seed`. Set the device once with `set_device`.
- Do not let the prose depend on output that varies between machines, such as paths, timings, the device or training
  results.
- The docs build reuses cached notebook outputs from the `docs-artifacts/<branch>` branch
  ([Documentation cache management](CACHE_MANAGEMENT.md)). Run `uv run nox -s docs -- clean` when a library change could
  alter an output.

## Links, roles and fences

### Roles

Write roles in the MyST spelling in Markdown and in notebook cells, and in the reST spelling in docstrings.

| Where | Spelling | Example |
| ----- | -------- | ------- |
| Markdown pages and notebook cells | MyST | `` {doc}`data_cleaning` ``, `` {py:class}`~dataeval_flow.Result` ``, `` {term}`Verdict` `` |
| Docstrings | reST | `` :class:`~dataeval_flow.Result` `` |

The Sphinx build runs with `--fail-on-warning`, so a `{doc}` or `{term}` role that cannot resolve fails the docs build.
The build suppresses warnings about unresolved Python references, so review checks a `{py:class}` target. Define a term
once, in the [Glossary](source/reference/glossary.md), and link it with `{term}`.

### Links

- Link to a page with `{doc}` and its source name, or with a relative `.md` path. Never link to the `.html` or `.ipynb`
  file.
- Use the exact title of the target page as the link text. When you retitle a page, search for the old title and update
  every link that repeats it.
- Heading anchors are generated to depth 4, and the docs build fails on a missing anchor. Prefer a link to a page over a
  link to a section. Links from outside the documentation to a section cannot be checked, and they break when you rename
  the heading.
- The `linkchecker` CI job checks the links in every Markdown page under `docs/source`, in `docs/CACHE_MANAGEMENT.md`
  and in the files at the root of the repository. It does not read notebook cells.

### Fences and admonitions

- Put the opening and the closing fence of a directive on lines of their own. Do not put text on the fence line.
- A fence closes only with a fence of the same length or longer. Put a code example that uses three backticks inside a
  directive that uses four.
- Use `:::{note}`, `:::{important}` or a backtick fence for admonitions.

## Add, rename or remove a page

### Add a page

1. Add the file name to the `toctree` of the matching section in `docs/source/tutorials/index.md` or
   `docs/source/how_to/index.md`, and add a row to the list table below it. The row has `` {doc}`Title <path>` `` and a
   one-sentence description.
2. Link the page from the question it answers in `docs/source/reference/index.md`, from the catalog entry of the step it
   teaches, and from the explanation it belongs to.
3. Add a line under `## Unreleased` in `CHANGELOG.md` for a page that a user reads. The changelog is written by hand and
   is not generated from commit subjects.

### Rename or remove a page

- Search every reference with `git grep -n <name>`, and update the toctree entry, the list row and every link in the
  same commit.
- A rename changes the published address. `README.md` links to published pages, and the `linkchecker` job checks those
  links against `docs/source`.
- Check the open merge requests for new links to the old name before you merge the rename.

## Write docstrings and descriptions

- `ruff` requires a docstring on every public module, class, function and method (`D1`, with the `pep257` convention).
  Start with a one-line summary. Write roles and code in the reST spelling, with double backticks around code.
- End the docstring of a config class that takes YAML with an `Example YAML::` block, as the existing config classes do.
- The `description` of a config field and of a step feeds `config/params.schema.json` and the config reference, so it
  follows the prose rules. A step's description is one sentence, and `tests/test_naming_conventions.py` checks that.
  Run `uv run nox -s schema` after you change one.

## Check your work

Run the checks that apply before you open a merge request.

| Command | What it checks |
| ------- | -------------- |
| `uv run nox -s lint` | `ruff`, `ruff format` and `codespell` |
| `uv run pytest tests/test_catalog_docs.py tests/test_naming_conventions.py tests/test_how_to_yaml.py` | The catalog pages, step names and descriptions, and the YAML in the how-tos |
| `npx --yes markdownlint-cli2 '*.md' 'docs/**/*.md' '#node_modules'` | The Markdown rules that CI enforces, such as 120 columns and one blank line between blocks. Run it after every change to a `.md` file. It needs Node, and `npx` fetches the tool on first use |
| `uv run nox -s schema` | `config/params.schema.json` after a change to a config description |
| `uv run nox -s docs` | Converts the notebooks, executes the ones that are not cached, runs their assertions and builds the site with `--fail-on-warning`, which catches broken links, anchors and toctree entries |
| `uv run nox -s docs -- skip` | The same build without executing notebooks, for a change to prose only |
| `uv run nox -s docs -- clean` | Clears the notebook cache and executes everything, for a library change that could alter outputs |

The markdownlint job in CI lints the files at the root of the repository and `docs/**/*.md`. The `linkchecker` job
checks the links of the root files, `docs/CACHE_MANAGEMENT.md` and `docs/source`.

What is not automated today: the voice and the plain wording of a sentence, whether bold marks a first mention, the
spelling of a notebook, the line width of a notebook cell, the order and the headings of a tutorial, the data-root tree,
the Next steps links, and every rule in the reports section. A reviewer checks them.

A notebook that needs more memory than your machine has cannot execute in your local docs build. Run its cells one at a
time instead, and rely on the CI docs job for the full run. Say in the merge request what you could and could not run.

## Review a prose-only change

A change that only rewrites wording must leave these unchanged in every notebook and page: code cells, link targets,
roles, numbers and the values in tables. Headings change only where this guide asks for it.

Compare the before and after trees to confirm it. Read the diff for sentences that gained a condition or lost a hedge.
"We recommend using image embeddings" becoming "Use image embeddings if you can" adds a condition and is a wording
change that alters the meaning.

## Open a documentation merge request

- Start the title with `[docs]`. [BRANCHING.md](../BRANCHING.md) explains the branch names and the release process.
- Make one deliverable for each merge request. Do not bundle unrelated documentation changes.
- Write each commit message as one line, `[docs] Subject`, with no body and no trailer.
- Leave the notebook cache alone. The docs build fetches it from the `docs-artifacts/<branch>` branch.
- List the checks you ran and the ones you could not run.
