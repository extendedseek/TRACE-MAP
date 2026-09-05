# Security policy

Do not post credentials, private model endpoints, unpublished checkpoints, or
sensitive logs in a public issue. Report a suspected vulnerability privately to
`nsadhan@ksu.edu.sa` with the affected revision, reproduction steps, and likely
impact.

The project loads optional third-party model code only when explicitly enabled.
Review model repositories before setting `trust_remote_code: true`, pin model
revisions, and keep network-facing inference services behind authentication.
