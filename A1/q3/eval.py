import sys

def parse_candidate_file(filepath):
    candidates = {}
    current_q = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('q #'):
                current_q = int(line.split('#')[1].strip())
            elif line.startswith('c #'):
                parts = line.split('#')[1].strip()
                if not parts:
                    candidates[current_q] = set()
                else:
                    candidates[current_q] = set(map(int, parts.split()))
    return candidates

def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate_quick.py <truth.dat> <candidates.txt>")
        sys.exit(1)

    # 1. Load Truth
    true_results = parse_candidate_file(sys.argv[1])
    
    # 2. Load Candidates
    candidate_results = parse_candidate_file(sys.argv[2])

    total_score = 0
    valid_submission = True
    queries_evaluated = 0

    print(f"{'Q_ID':<5} | {'True':<5} | {'Cand':<5} | {'Prec':<5}")
    print("-" * 30)

    for q_idx in sorted(true_results.keys()):
        R_q = true_results[q_idx]
        C_q = candidate_results.get(q_idx, set())
        
        missing = R_q - C_q
        if missing:
            print(f"FAIL: Q{q_idx} missed {len(missing)}")
            valid_submission = False
        
        if len(C_q) == 0:
            score = 1.0 if len(R_q) == 0 else 0.0
        else:
            score = len(R_q) / len(C_q)
            
        print(f"{q_idx:<5} | {len(R_q):<5} | {len(C_q):<5} | {score:.4f}")
        total_score += score
        queries_evaluated += 1

    avg = total_score / queries_evaluated if queries_evaluated else 0
    print("="*30)
    print(f"AVG PRECISION: {avg:.5f}")
    if not valid_submission:
        print("STATUS: INVALID (False Negatives)")

if __name__ == "__main__":
    main()