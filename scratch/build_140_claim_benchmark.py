import json
import os

with open("benchmark/eval_dataset.json", "r", encoding="utf-8") as f:
    existing_110 = json.load(f)

new_claims_30 = [
    # Book 3: The Hound of the Baskervilles (IDs 111-125)
    {
        "id": 111,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Sherlock Holmes is an English consulting detective residing at 221B Baker Street in London who investigates the mysterious death of Sir Charles Baskerville.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Sherlock Holmes is a consulting detective in London investigating the Baskerville case."
    },
    {
        "id": 112,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Dr. James Mortimer consulted Sherlock Holmes and presented the ancient manuscript describing the legendary curse of the Baskerville family.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Dr. Mortimer presented the 1742 Baskerville manuscript to Holmes and Watson."
    },
    {
        "id": 113,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Sir Henry Baskerville traveled from North America to claim his inheritance as the rightful heir to the Baskerville estate in Devonshire.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Sir Henry came from Canada/USA to inherit Baskerville Hall."
    },
    {
        "id": 114,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Sherlock Holmes was a former Scotland Yard inspector who retired in 1870 to manage Baskerville Hall as a paid bailiff.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Holmes was never a Scotland Yard inspector or bailiff for Baskerville Hall."
    },
    {
        "id": 115,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Sir Henry Baskerville was murdered on the London Underground by Dr. Mortimer before he could ever visit Baskerville Hall.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Sir Henry was not murdered in London; he survived and traveled to Dartmoor."
    },
    {
        "id": 116,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Dr. John Watson invested twenty pounds in an East London glass manufacturing company in 1884 upon the recommendation of an old army comrade.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned private investment."
    },
    {
        "id": 117,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Sir Henry Baskerville learned classical Spanish guitar during his agricultural travels across Western Canada.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned hobby."
    },
    {
        "id": 118,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Dr. Watson accompanied Sir Henry Baskerville to Dartmoor while Sherlock Holmes secretly stayed in a prehistoric stone hut on the moor to conduct independent surveillance of Jack Stapleton.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Holmes secretly hid on the moor in a stone hut while Watson protected Sir Henry."
    },
    {
        "id": 119,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Jack Stapleton was secretly the son of Rodger Baskerville who used a hound coated in phosphorus to frighten Sir Charles Baskerville to death.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Stapleton was Rodger Baskerville's son and used a phosphorus-painted hound."
    },
    {
        "id": 120,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Jack Stapleton was the loyal younger brother of Sir Henry Baskerville who saved Sherlock Holmes from drowning in the Grimpen Mire and became Baron of Devonshire.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Stapleton was a villainous cousin who drowned in the Grimpen Mire."
    },
    {
        "id": 121,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Dr. Watson secretly poisoned Sir Charles Baskerville with cyanide on the instructions of Inspector Lestrade to confiscate the moorlands for the British Crown.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Watson is loyal and never poisoned Sir Charles."
    },
    {
        "id": 122,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Barrymore the butler kept a collection of rare botanical fern drawings in the attic of Baskerville Hall which he traded with local Devonshire herbalists.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned hobby."
    },
    {
        "id": 123,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Mrs. Stapleton attended a finishing school in Brussels during her youth, where she specialized in watercolor landscape painting and French literature.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned backstory."
    },
    {
        "id": 124,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long_paragraph",
        "user_input": "Sherlock Holmes sent Dr. Watson to Devonshire to protect Sir Henry Baskerville from the mortal danger looming over Baskerville Hall, while Holmes secretly established himself in an abandoned Neolithic hut on the misty moor to observe events undetected. Through Watson's detailed diary reports, Holmes unmasked Jack Stapleton as a fraudulent naturalist living at Merripit House who was actually a rogue Baskerville claimant scheming to murder the legitimate heirs. Utilizing a massive hound painted with glowing phosphorus, Stapleton ambushed Sir Henry on the moor, but Holmes and Lestrade intervened in time, shooting the beast and driving the fleeing Stapleton to his death in the treacherous bogs of the Great Grimpen Mire.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Canonical resolution of The Hound of the Baskervilles."
    },
    {
        "id": 125,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long_paragraph",
        "user_input": "Sherlock Holmes concluded that the legendary hound was merely an optical illusion engineered by Dr. Watson to drive Sir Henry Baskerville insane and seize the ancestral fortune. After arresting Dr. Watson and Dr. Mortimer at the Princetown railway station, Holmes officially purchased Baskerville Hall and retired permanently to breed bloodhounds in Devonshire, renouncing his London detective practice to become the High Sheriff of Dartmoor alongside the newly crowned Baron Stapleton.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Fabricated absurd conspiracy contradicting canonical events."
    },

    # Book 4: Dracula (IDs 126-140)
    {
        "id": 126,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Jonathan Harker is an English solicitor's clerk who traveled to Transylvania to assist Count Dracula with purchasing the Carfax estate in England.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Jonathan Harker traveled to Castle Dracula for real estate transactions."
    },
    {
        "id": 127,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Professor Abraham Van Helsing is a Dutch physician and specialist in obscure diseases who led the investigation against Count Dracula.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Van Helsing is a Dutch polymath and vampire hunter."
    },
    {
        "id": 128,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Mina Murray transcribed phonograph diaries and letters to reconstruct Count Dracula's travel movements across England.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Mina Harker transcribed and compiled all records."
    },
    {
        "id": 129,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Count Dracula was an English merchant from Manchester who traded wool and spices and never visited Transylvania.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Dracula is a Transylvanian vampire nobleman."
    },
    {
        "id": 130,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Jonathan Harker murdered Professor Van Helsing with a silver sword in Munich after discovering Van Helsing was a vampire.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Van Helsing survived and was never killed by Harker."
    },
    {
        "id": 131,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Jonathan Harker collected antique brass postage scales as a private hobby during his legal clerkship in Exeter.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned private hobby."
    },
    {
        "id": 132,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Dr. John Seward studied the migration patterns of North Sea seabirds while vacationing at Whitby in his early twenties.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned backstory."
    },
    {
        "id": 133,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Count Dracula chartered the Russian schooner Demeter from Varna to Whitby, carrying fifty boxes of earth and draining the crew until the derelict ship crashed in a storm.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "The Demeter voyage carried Dracula and boxes of earth to Whitby."
    },
    {
        "id": 134,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Following the vampiric transformation of Lucy Westenra in London, Professor Van Helsing and Arthur Holmwood staked her heart to give her peace.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Van Helsing and Holmwood drove a stake through Lucy's heart."
    },
    {
        "id": 135,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Count Dracula was formally welcomed at Buckingham Palace by Queen Victoria and appointed Lord Mayor of London after presenting Van Helsing with the Golden Cross.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Dracula was never welcomed by Queen Victoria or made Lord Mayor."
    },
    {
        "id": 136,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Mina Harker betrayed Jonathan to marry Count Dracula at Carfax Abbey, becoming the immortal Empress of Transylvania while Van Helsing fled to Holland.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Mina resisted Dracula and Dracula was destroyed."
    },
    {
        "id": 137,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Renfield worked as an apprentice watchmaker in Leeds before his mental deterioration and confinement in Dr. Seward's asylum.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned early profession."
    },
    {
        "id": 138,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Quincey Morris owned extensive silver mining shares in Nevada, which he managed through a broker in San Francisco before traveling to Europe.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned investment."
    },
    {
        "id": 139,
        "book": "Dracula",
        "claim_type": "long_paragraph",
        "user_input": "Jonathan Harker was held captive in Castle Dracula in Transylvania, witnessing horrifying supernatural phenomena before making a daring escape down the castle walls into a convent hospital in Budapest. Reunited with his devoted wife Mina, Jonathan joined forces with Dr. John Seward, Arthur Holmwood, Quincey Morris, and Professor Abraham Van Helsing. Utilizing crucifixes, sacred wafers, and garlic, the hunters tracked the vampire count across London and pursued him back to Transylvania, where Jonathan Harker slit Dracula's throat with a kukri knife while Quincey Morris drove a bowie knife into the monster's heart as the sun set over the Carpathian Mountains.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Canonical climax and resolution of Dracula."
    },
    {
        "id": 140,
        "book": "Dracula",
        "claim_type": "long_paragraph",
        "user_input": "Count Dracula successfully turned Jonathan Harker and Professor Van Helsing into subservient vampire generals, who then led an army of the undead across the English Channel to conquer London in 1897. After converting Mina Harker and Lucy Westenra into vampire queens at St. Paul's Cathedral, Dracula signed a formal peace treaty with Prime Minister Lord Salisbury, establishing Transylvanian sovereign rule over the British Empire and appointing Renfield as Chancellor of the Exchequer.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Fabricated alternate history contradicting canonical Dracula."
    }
]

all_140 = existing_110 + new_claims_30

# Write to benchmark/eval_dataset.json and benchmark/eval_dataset_140.json
with open("benchmark/eval_dataset.json", "w", encoding="utf-8") as f:
    json.dump(all_140, f, ensure_ascii=False, indent=2)

with open("benchmark/eval_dataset_140.json", "w", encoding="utf-8") as f:
    json.dump(all_140, f, ensure_ascii=False, indent=2)

with open("Data/eval_dataset.json", "w", encoding="utf-8") as f:
    json.dump(all_140, f, ensure_ascii=False, indent=2)

print(f"Successfully constructed 140-Claim Multi-Novel Benchmark Suite across 4 Books!")
print(f"Total Claims: {len(all_140)}")

book_counts = {}
verdict_counts = {}
type_counts = {}
for c in all_140:
    b = c["book"]
    v = c["ground_truth_verdict"]
    ct = c.get("claim_type", "short")
    book_counts[b] = book_counts.get(b, 0) + 1
    verdict_counts[v] = verdict_counts.get(v, 0) + 1
    type_counts[ct] = type_counts.get(ct, 0) + 1

print("\nBy Book:")
for b, cnt in book_counts.items():
    print(f"  - {b}: {cnt} claims")

print("\nBy Verdict Class:")
for v, cnt in verdict_counts.items():
    print(f"  - {v}: {cnt} claims")

print("\nBy Granularity:")
for ct, cnt in type_counts.items():
    print(f"  - {ct}: {cnt} claims")
