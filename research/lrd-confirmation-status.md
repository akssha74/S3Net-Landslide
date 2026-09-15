# LRD external-analysis status

The original protocol and executable code were committed locally before
validation/protected optical access, and the checkpoint authorization,
protected-fetch log, and outputs form a consistent hash chain. No public or
third-party timestamp predating access exists. The run is therefore described
as **internally precommitted but externally time-unverified**, not as a
third-party-verifiable preregistration.

The released EID split also does not equal trigger-level independence:
development EID PH0003 and protected EIDs PH0001/PH0004 share the Philippines
2022-04-10 trigger family. Four protected EIDs remain non-overlapping under the
minimum country-plus-trigger-date family rule.

Finally, 84.1% of selected protected crops are empty. Under the frozen
empty/empty boundary score of one, the mean boundary effect is -24.10 points;
with empty/empty scored zero it is -0.15, and common-threshold/pooled variants
are near zero or mixed. Validation maximum F1 ranges from 0.008 to 0.151.

The frozen arithmetic remains reproducible and all four frozen criteria fail,
but the external result is classified as **failed/indeterminate sensitivity
evidence**, not robust confirmation or rejection of transfer. No test-driven
retuning was performed.
