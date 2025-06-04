try:
    from .scm_sampler import register_scm_sampler
    register_scm_sampler()
    print("SCM sampler registered successfully")
except Exception as e:
    print(f"Failed to register SCM sampler: {e}") 