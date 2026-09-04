# Authentication token expiry incident
Users were logged out early because access-token expiry was calculated in milliseconds while the verifier expected seconds. Token validation itself remained healthy.
