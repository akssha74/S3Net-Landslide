# Semantic-identifiability certificate

## Purpose

A physical or mechanism contrast can depend on a data schema that the release
does not uniquely identify. Selecting one convenient schema hides that
dependence. The certificate propagates every metadata-consistent schema into the
signed contrast.

## Definition

For finite admissible schema set `O` and signed contrast `delta(o)`, define the
semantic envelope as:

`[min(delta(o) for o in O), max(delta(o) for o in O)]`.

The direction is:

- `positive-identified` if the lower bound is above zero;
- `negative-identified` if the upper bound is below zero;
- `direction-unidentified` if the envelope contains zero.

This is a finite sensitivity result over descriptive effects. It is not a
confidence interval, a population statement, or proof that the candidate-schema
set is exhaustive.

The certificate was formalized after the dual-order runs. It reorganizes the
frozen contrasts without selecting or fitting a new model and receives no
prospective-confirmation credit.

## HR-GLDD binding

The admissible set is `{RGBN, BGRN}` because Green/NIR are fixed while released
columns 0/2 are not authoritatively bound to Red/Blue. The certificate covers
four mechanism contrasts already present in the complete
factorial surface:

1. candidate-index input minus raw-edge input on F1;
2. equal-mass index modulation minus plain boundary weighting on F1;
3. the same loss contrast on boundary F1;
4. plain boundary weighting minus base loss on F1.

The generated JSON binds every contrast to the dual-order result artifact and
records the source SHA-256. An independent test recomputes the values from the
two order-specific run summaries.
