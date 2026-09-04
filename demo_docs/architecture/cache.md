# Cache architecture
Read-heavy API endpoints use a shared cache before the relational database. Cache keys include tenant and resource identifiers.
