import sys
sys.path.append("Pipeline")
from claimExtraction import extract_atomic_claims

c7 = "Abbé Faria was an Italian priest imprisoned in the Château d'If who revealed the location of the hidden treasure on the island of Monte Cristo to Dantès."
c8 = "Danglars was the purser of the Pharaon who harbored jealousy toward Edmond Dantès' promotion to captain."
c9 = "Major MacNabb is the captain of the yacht Duncan who hooked and landed the hammerhead shark in the North Channel."

print("Claim 7 Atomic:", extract_atomic_claims(c7))
print("Claim 8 Atomic:", extract_atomic_claims(c8))
print("Claim 9 Atomic:", extract_atomic_claims(c9))
