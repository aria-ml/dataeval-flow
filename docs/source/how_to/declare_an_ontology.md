# Declare an ontology

The `label-space` workflow judges a dataset's labels against a declared {term}`ontology <Ontology>` — the sanctioned
label space. It can name a missing class because the ontology says that class was supposed to exist. It is the only
workflow that judges labels against an ontology: `data-coverage` refuses `ontology:`, so run a `label-space` entry on
the same source beside it. This guide covers declaring an ontology inline, loading one from an RDF file, and the
findings `label-space` makes from it.

## Used in these tutorials

- {doc}`Assess dataset coverage <../notebooks/data_coverage>`

## Why counting labels is not enough

A class-balance worklist built from the dataset's own `index2label` is circular: it can only name classes the dataset
already declares. A class that was never collected has no label, no count, and no row in the report. Declaring the
label space externally is what breaks the circle, and `label-space` requires one: `ontology:` has no default.

## Option 1: inline hierarchy

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
    type: label-space
    ontology:
      postal_char:
        digit:
          low: ["0", "1", "2", "3", "4"]
          high: ["5", "6", "7", "8", "9"]
        letter:
          vowel: ["A", "E", "I", "O", "U"]
```

## Option 2: a versioned RDF artifact

For a label space that is shared across datasets, teams, or programs, keep it in a file and reference it by path:

```yaml
workflows:
  - name: vocab_check
    type: label-space
    ontology: config/taxonomy.ttl
```

Supported serializations are inferred from the suffix: `.ttl` (Turtle), `.rdf` / `.owl` / `.xml` (RDF/XML), `.nt`
(N-Triples), and `.jsonld` / `.json` (JSON-LD). Relative paths resolve against the data root.

This path needs `rdflib`, which arrives with the `ontology` extra:

```bash
pip install "dataeval-flow[ontology]"
```

An RDF artifact carries what an inline mapping cannot: synonyms, definitions, and stable concept identifiers that
survive a class being renamed. A config-declared concept (see Option 3) carries the same three fields without a
file. Reach for an RDF artifact when the label space is authored or reviewed outside this config.

## Option 3: a shared `ontologies:` block

A label space is a decision, not a per-workflow setting. Two workflows reading different ontologies produce
worklists you cannot compare. Define it once under `ontologies:` and reference it by name, the same way `datasets`,
`views`, `sources`, `extractors`, and `metadata` work:

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
  - name: audit
    type: label-space
    ontology: vehicles
  - name: audit_holdout
    type: label-space
    ontology: vehicles        # same pool entry, same vocabulary
```

Use `concepts:` to add concepts on top of `source`, or omit `source` and declare the whole label space in config.
Declare a concept to keep a dataset class the artifact omits. Give each declared concept the dataset's own spelling
under `synonyms`: alignment matches on labels and synonyms, and a concept without the dataset's spelling will not
match that class.

A declared concept replaces one the artifact defines under the same id. Replacement is total: restate `parents` on
it, or the concept becomes a root and detaches from the hierarchy.

A workflow's `ontology:` value is read as a name in the pool first, and as a path second, so a config that already
names a file keeps working unchanged. A string that matches both a pool entry and a readable file is refused.
Rename one of them.

## Set expected class shares

By default every sanctioned class is held to a uniform share of the dataset. When some classes are legitimately rarer
than others, give them explicit floors with `expected` — a mapping of class name to its minimum expected
share as a fraction in `[0, 1]`:

```yaml
    expected:
      face_shield: 0.05
      goggles: 0.02
```

Named classes use their floor as the collection target instead of the uniform share, and a dataset below the floor is
reported as a violation. Classes not named keep the uniform target. A name that resolves to no concept, or to several,
is ignored and noted in the result.

## Lint the label names

`label_pattern` is a regex every concept label should match. It catches a vocabulary that has drifted into mixed
conventions:

```yaml
    label_pattern: '^[a-z0-9_]+$'   # lowercase_snake_case
```

Labels that fail are reported in the ontology's structure.

## What `label-space` finds

`label-space` runs four evaluators, each followed by the check that judges it:

- Leaf coverage and the worklist, by `leaf-coverage`: how many sanctioned leaf concepts have examples, what to collect,
  the wholly empty branches, and the `expected` shares not met.
- Conformance, by `label-conformance`: which class names resolve to exactly one concept. It warns on an unmatched or
  an ambiguous name.
- Alignment, by `mergeability`: whether the dataset's classes carry over to the ontology, with the `Relabel` stanza to
  paste into a view that conforms it.
- Structure, by `ontology-structure`: the ontology's size, depth and naming. It warns on a label several concepts
  share.

Two of the checks have thresholds, set under `health_thresholds` and keyed by check type. `null` turns a threshold
off. The values below are the defaults:

```yaml
    health_thresholds:
      leaf-coverage: {coverage: 0.9, empty_branches: 0}
      label-conformance: {warning: 0}
```

| Check | Threshold | Default | Meaning |
| --- | --- | --- | --- |
| `leaf-coverage` | `coverage` | `0.9` | Minimum fraction of sanctioned leaf concepts with any examples |
| `leaf-coverage` | `empty_branches` | `0` | Wholly unpopulated branches tolerated before warning |
| `label-conformance` | `warning` | `0` | Class names that may fail to resolve to a concept |

Leaf coverage and empty branches catch the class you never collected. Unmatched names catch the opposite problem — a
label in the data that the sanctioned vocabulary does not contain, which is usually a typo, a stale name, or a class
someone added without updating the taxonomy.

## The label space digest

The result's `label_space_digest` is the alignment's: the value a dataset conformed by the alignment's `Relabel`
stanza carries. Two results with the same digest judged the same label space. If a source's `Relabel` already recorded
a label space, that record's digest is used instead.

## Related material

- [Dataset Coverage](../concepts/Coverage.md) — the label-space and embedding-space axes coverage measures
- [DataEval Ontology explanation](https://dataeval.readthedocs.io/en/latest/concepts/Ontology.html) — the
  authoritative treatment of ontologies and the reconciliation, alignment, and validation operations over them
- {doc}`API Reference <../reference/autoapi/dataeval_flow/index>` — every field on `LabelSpaceConfig` and
  `LabelSpaceThresholds`
