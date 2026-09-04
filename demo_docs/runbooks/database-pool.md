# Database pool recovery
When connection pool utilization is saturated, confirm active sessions, reduce leaked connections, and restart only the affected application workers. Do not increase the pool without checking the database session limit.
