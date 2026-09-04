# Database architecture
Application workers borrow database connections from a bounded pool. The pool protects the primary database from unbounded concurrent sessions.
