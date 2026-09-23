from ir_measures import alpha_nDCG, StRecall

try:
    from .rac_eval import main
except ImportError:
    from rac_eval import main

if __name__ == "__main__":
    MAX_CUTOFF = 20
    main(
        [StRecall@k for k in range(1, MAX_CUTOFF + 1)] + \
        [alpha_nDCG@k for k in range(1, MAX_CUTOFF + 1)]
    )
