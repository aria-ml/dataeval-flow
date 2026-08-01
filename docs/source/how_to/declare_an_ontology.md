# Declare an ontology

The `data-coverage` workflow can only name a missing class if something tells it that class was supposed to exist.
That something is an {term}`ontology <Ontology>` — the sanctioned label space. This guide covers declaring one inline,
loading one from an RDF file, and the checks that only run once you have.

## Used in these tutorials

- {doc}`Assess dataset coverage <../notebooks/data_coverage>`

## Why counting labels is not enough

Without an ontology the workflow synthesizes a flat one from the dataset's own `index2label`. That is enough for a
class-balance worklist, but it is circular: it can only name classes the dataset already declares. A class that was
never collected has no label, no count, and no row in the report. Declaring the label space externally is what breaks
the circle.

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
  - name: coverage_check
    type: data-coverage
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
  - name: coverage_check
    type: data-coverage
    ontology: config/taxonomy.ttl
```

Supported serializations are inferred from the suffix: `.ttl` (Turtle), `.rdf` / `.owl` / `.xml` (RDF/XML), `.nt`
(N-Triples), and `.jsonld` / `.json` (JSON-LD). Relative paths resolve against the data root.

This path needs `rdflib`, which ships in DataEval's ontology extra:

```bash
pip install "dataeval[ontology]"
```

An RDF artifact also carries what an inline mapping cannot: synonyms, definitions, and stable concept identifiers that
survive a class being renamed. Prefer it for anything you intend to keep.

## Set expected class shares

By default every sanctioned class is held to a uniform share of the dataset. When some classes are legitimately rarer
than others, give them explicit floors with `ontology_expected` — a mapping of class name to its minimum expected
share, as a fraction in `[0, 1]`:

```yaml
    ontology_expected:
      face_shield: 0.05
      goggles: 0.02
```

Named classes use their floor as the collection target instead of the uniform share, and a dataset below the floor is
reported as a violation. Classes not named keep the uniform target.

## Lint the label names

`ontology_label_pattern` is a regex every concept label must match — useful for catching a vocabulary that has drifted
into mixed conventions:

```yaml
    ontology_label_pattern: '^[a-z0-9_]+$'   # lowercase_snake_case
```

Labels that fail are reported. The pattern is ignored when the ontology is synthesized.

## What a configured ontology unlocks

Three findings — and the three health thresholds that govern them — apply **only** when an ontology is configured.
Against a synthesized ontology they are vacuous by construction, so they never fire.

| Finding | Threshold | Default | Meaning |
| --- | --- | --- | --- |
| Leaf coverage | `leaf_coverage` | `0.9` | Minimum fraction of sanctioned leaf concepts with any examples |
| Dark branches | `dark_branch_count` | `0` | Wholly unpopulated branches tolerated before warning |
| Unmatched classes | `unmatched_class_count` | `0` | Class names that may fail to resolve to a concept |

```yaml
    health_thresholds:
      leaf_coverage: 0.9
      dark_branch_count: 0
      unmatched_class_count: 0
```

Leaf coverage and dark branches catch the class you never collected. Unmatched classes catch the opposite problem — a
label in the data that the sanctioned vocabulary does not contain, which is usually a typo, a stale name, or a class
someone added without updating the taxonomy.

## Related material

- [Dataset Coverage](../concepts/Coverage.md) — the label-space and embedding-space axes coverage measures
- [DataEval Ontology explanation](https://dataeval.readthedocs.io/en/latest/concepts/Ontology.html) — the
  authoritative treatment of ontologies and the reconciliation, alignment, and validation operations over them
- {doc}`API Reference <../reference/autoapi/dataeval_flow/index>` — every field on `DataCoverageParameters` and
  `DataCoverageHealthThresholds`
