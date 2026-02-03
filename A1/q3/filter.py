import sys
import numpy as np

if __name__ == "__main__":
    db = np.load(sys.argv[1])
    q = np.load(sys.argv[2])
    out = sys.argv[3]
    
    f = open(out, 'w')
    for i in range(len(q)):
        qvec = q[i]
        and_result = np.bitwise_and(db, qvec)
        matches = np.all(and_result == qvec, axis=1)
        cands = np.where(matches)[0]
        
        f.write(f"q # {i}\n")
        f.write(f"c # {' '.join(map(str, cands))}\n")
    f.close()