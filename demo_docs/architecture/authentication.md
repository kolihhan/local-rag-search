# Authentication architecture
The identity service issues signed access tokens. API services validate the signature, audience, issuer, and expiry locally using cached public keys.
