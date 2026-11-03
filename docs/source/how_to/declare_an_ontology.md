# Declare an ontology

The [`taxonomy`](../reference/presets.md#taxonomy) workflow judges a dataset's labels against a declared
{term}`ontology <Ontology>`, the sanctioned label space. The ontology says which classes were supposed to exist, so
`taxonomy` can name a class that was never collected. A class shortfall list that
[`class-shortfall`](../reference/checks.md#class-shortfall) builds from the dataset's own `index2label` cannot. It can
only name classes the dataset already declares, and a class that was never collected has no label, no count and no row
in the report. [Dataset Coverage](../concepts/Coverage.md) explains the gap. The
[DataEval Ontology explanation](https://dataeval.readthedocs.io/en/latest/concepts/Ontology.html) covers ontologies and
the reconciliation, alignment and validation operations over them.

`taxonomy` requires an ontology, because `ontology:` has no default. It is the only workflow that judges labels against
one: [`scope`](../reference/presets.md#scope) refuses `ontology:`, so run a `taxonomy` entry on the same source beside
it. This guide declares an ontology inline, loads one from an RDF file, shares one across workflows, and reads the
findings `taxonomy` makes from it.

## Declare an inline hierarchy

For a small, stable label space, write the hierarchy directly into the workflow config as a nested mapping of concept
to children. Leaves are lists.

```python
import string

postal_ontology = {
    "postal_char": {
        "digit": {
            "low": [str(d) for d in range(5)],
            "high": [str(d) for d in range(5, 10)],
        },
        "letter": {
            "vowel": list("AEIOU"),
            "consonant": [c for c in string.ascii_uppercase if c not in "AEIOU"],
        },
    }
}
```

The same structure in YAML:

```yaml
workflows:
  - name: vocab_check
    type: taxonomy
    ontology:
      postal_char:
        digit:
          low: ["0", "1", "2", "3", "4"]
          high: ["5", "6", "7", "8", "9"]
        letter:
          vowel: ["A", "E", "I", "O", "U"]
```

## Load a versioned RDF artifact

For a label space shared across datasets, teams or programs, keep it in a file and reference it by path:

```yaml
workflows:
  - name: vocab_check
    type: taxonomy
    ontology: config/taxonomy.ttl
```

Flow infers the serialization from the suffix: `.ttl` (Turtle), `.rdf` / `.owl` / `.xml` (RDF/XML), `.nt` (N-Triples),
and `.jsonld` / `.json` (JSON-LD). Relative paths resolve against the data root.

This path needs `rdflib`, which arrives with the `ontology` extra:

```bash
pip install "dataeval-flow[ontology]"
```

An RDF artifact carries what an inline mapping cannot: synonyms, definitions, and stable concept identifiers that
survive a class being renamed. A config-declared concept (see the "Share an ontology across workflows" section) carries
the same three fields without a file. Use an RDF artifact when the label space is authored or reviewed outside this
config.

## Share an ontology across workflows

Two workflows that read different ontologies produce worklists you cannot compare. Define the label space once under
`ontologies:` and reference it by name, the same way `datasets`, `views`, `sources`, `extractors` and `metadata` work:

```yaml
ontologies:
  - name: vehicles
    source: config/label_ontology.jsonld
    concepts:
      - id: http://example.org/cv#FreightCar
        label: Freight Car
        synonyms: [freight_car, freight car]
        parents: [http://example.org/cv#LandVehicle]

workflows:
  - name: vehicle-labels
    type: taxonomy
    ontology: vehicles
  - name: vehicle-holdout
    type: taxonomy
    ontology: vehicles        # same pool entry, same vocabulary
```

Use `concepts:` to add concepts on top of `source`, or omit `source` and declare the whole label space in config.
Declare a concept to keep a dataset class the artifact omits. Give each declared concept the dataset's own spelling
under `synonyms`: alignment matches on labels and synonyms, and a concept without the dataset's spelling will not
match that class.

A declared concept replaces one the artifact defines under the same id. Replacement is total: restate `parents` on
it, or the concept becomes a root and detaches from the hierarchy.

A workflow's `ontology:` value is read as a name in the pool first, and as a path second, so a config that already
names a file keeps working unchanged. A string that matches both a pool entry and a readable file is refused. Rename
one of them.

## Set expected class shares

By default every sanctioned class is held to a uniform share of the dataset. When some classes are legitimately rarer
than others, give them explicit floors with `representation.expected` (see
[`representation`](../reference/evaluators.md#representation)). It is a mapping of class name to its minimum expected
share as a fraction in `[0, 1]`:

```yaml
    representation:
      expected:
        face_shield: 0.05
        goggles: 0.02
```

Named classes use their floor as the collection target instead of the uniform share, and a dataset below the floor is
reported as a violation. Classes not named keep the uniform target. A name that resolves to no concept, or to several,
is ignored and noted in the result.

## Lint the label names

`ontology-validation.label_pattern` (see [`ontology-validation`](../reference/evaluators.md#ontology-validation)) is a
regex every concept label should match. It catches a vocabulary that has
drifted into mixed conventions:

```yaml
    ontology-validation:
      label_pattern: '^[a-z0-9_]+$'   # lowercase_snake_case
```

Labels that fail are reported in the ontology's structure.

## Read what `taxonomy` finds

`taxonomy` runs four evaluators, each followed by the check that judges it:

- Leaf coverage and the worklist, by [`leaf-coverage`](../reference/checks.md#leaf-coverage): how many sanctioned leaf
  concepts have examples, what to collect, the wholly empty branches, and the `expected` shares not met.
- Conformance, by [`label-conformance`](../reference/checks.md#label-conformance): which class names resolve to exactly
  one concept. It warns on an unmatched or an ambiguous name.
- Alignment, by [`label-mergeability`](../reference/checks.md#label-mergeability): whether the dataset's classes carry
  over to the ontology, with the `Relabel` stanza to paste into a view that conforms it.
- Structure, by [`ontology-structure`](../reference/checks.md#ontology-structure): the ontology's size, depth and
  naming. It warns on a label several concepts share.

## Set the check thresholds

Two of the checks have thresholds. Set them under `checks`, keyed by check type. `leaf-coverage` takes `coverage`, the
least fraction of sanctioned leaf concepts with any examples, and `empty_branches`, the wholly unpopulated branches
tolerated before it warns. `label-conformance` takes `warning`, the class names that may fail to resolve to a concept.
`null` turns a threshold off. The values below are the defaults, which the
[Preset Catalog](../reference/presets.md#taxonomy) lists with the rest of the preset's settings:

```yaml
    checks:
      leaf-coverage: {coverage: 0.9, empty_branches: 0}
      label-conformance: {warning: 0}
```

Leaf coverage and empty branches catch the class you never collected. Unmatched names catch the opposite problem: a
label in the data that the sanctioned vocabulary does not contain. It is usually a typo, a stale name, or a class
someone added without updating the taxonomy.

## Check which label space a result judged

The result's `label_space_digest` is the alignment's: the value a dataset conformed by the alignment's `Relabel`
stanza carries. Two results with the same digest judged the same label space. If a source's `Relabel` already recorded
a label space, that record's digest is used instead.

## Next steps

- [Assess dataset coverage](../notebooks/data_coverage.py) — Run `taxonomy` beside `scope` on one source to name the
  classes a dataset is missing.
- [Dataset Coverage](../concepts/Coverage.md) — The label-space and embedding-space axes that coverage measures.
- [Merge sources into one dataset](merge_sources.md) — Conform two datasets to a shared vocabulary with `Relabel` and
  combine them.
