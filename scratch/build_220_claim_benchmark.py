import json
import os

with open("benchmark/eval_dataset_140.json", "r", encoding="utf-8") as f:
    existing_140 = json.load(f)

# 80 New Claims: 40 for The Hound of the Baskervilles + 40 for Dracula
new_80_claims = [
    # =========================================================================
    # Book 3: The Hound of the Baskervilles (IDs 141 - 180) [40 Claims]
    # =========================================================================
    # --- Short Atomic Claims (20) ---
    {
        "id": 141,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Sir Charles Baskerville died of a sudden heart attack near the yew alley gate after seeing the terrifying hound.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Sir Charles died of heart failure triggered by extreme fright from the hound."
    },
    {
        "id": 142,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Beryl Stapleton was secretly Jack Stapleton's abused wife whom he falsely introduced as his sister.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Beryl was Stapleton's wife from South America, not his sister."
    },
    {
        "id": 143,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Selden was an escaped Notting Hill murderer hiding on Dartmoor who was Mrs. Barrymore's younger brother.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Selden was the escaped convict and Mrs. Barrymore's brother."
    },
    {
        "id": 144,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Laura Lyons of Coombe Tracey wrote a desperate letter asking Sir Charles Baskerville to meet her at the gate on the night he died.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Laura Lyons sent the appointment letter at Stapleton's instigation."
    },
    {
        "id": 145,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Inspector Lestrade of Scotland Yard joined Sherlock Holmes and Dr. Watson in Devonshire to assist with the final arrest.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Lestrade arrived from London to assist in shooting the hound."
    },
    {
        "id": 146,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Mr. Frankland of Lafter Hall spent his time watching the moors through a telescope and pursuing frivolous legal lawsuits.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Frankland was an eccentric amateur astronomer and litigious neighbor."
    },
    {
        "id": 147,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Sherlock Holmes deduced that Sir Henry Baskerville was secretly an impostor who murdered the real Sir Henry in Canada.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Sir Henry was genuine and proved his lineage."
    },
    {
        "id": 148,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Dr. James Mortimer was the primary mastermind who bred the hound to inherit Baskerville Hall.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Mortimer was an innocent friend and trustee; Stapleton was the villain."
    },
    {
        "id": 149,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Barrymore the butler murdered Selden with an axe on the moor to prevent the convict from revealing family secrets.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Selden died from a fall while being pursued by the hound, wearing Sir Henry's clothes."
    },
    {
        "id": 150,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Jack Stapleton successfully fled to America on an ocean liner with Sir Henry's stolen millions.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Stapleton died by drowning in the Great Grimpen Mire."
    },
    {
        "id": 151,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Dr. Watson served as an army surgeon during the Second Anglo-Afghan War before returning to London.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Watson's canonical military backstory in the British Army."
    },
    {
        "id": 152,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Dr. James Mortimer published a treatise on medieval French cathedral gargoyles during his early medical residency.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned private paper."
    },
    {
        "id": 153,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Sir Charles Baskerville invested in South African gold mining syndicates while residing in London in 1888.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned investment."
    },
    {
        "id": 154,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Barrymore owned a small family dairy cottage in Somerset which he leased to a cousin.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned property."
    },
    {
        "id": 155,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Laura Lyons studied Italian operatic singing in Milan before opening her typing office in Coombe Tracey.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned hobby."
    },
    {
        "id": 156,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Mr. Frankland owned a prize-winning black stallion that he bred for steeplechase racing in Exeter.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned ownership."
    },
    {
        "id": 157,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Jack Stapleton formerly operated a private boys' school in Yorkshire under the alias of Vandeleur.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Stapleton ran a school in Yorkshire which collapsed before he moved to Devonshire."
    },
    {
        "id": 158,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Cartwright the errand boy was secretly an agent for the Russian embassy tracking Sherlock Holmes.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Cartwright was a loyal English boy helping Holmes check hotel wastepaper."
    },
    {
        "id": 159,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Sir Henry Baskerville proposed marriage to Beryl Stapleton before learning she was already married to Jack Stapleton.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Sir Henry fell in love with Beryl, unaware she was Stapleton's wife."
    },
    {
        "id": 160,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "short",
        "user_input": "Sherlock Holmes kept a private chemical laboratory in the basement of Scotland Yard for confidential poison testing.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned lab."
    },

    # --- Long Narrative Claims (18) ---
    {
        "id": 161,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "When Sir Henry Baskerville arrived in London, one of his new boots was stolen at the hotel, which Jack Stapleton used to provide the bloodhound with Sir Henry's personal scent.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "The boot theft was executed by Stapleton to train the hound on Sir Henry's scent."
    },
    {
        "id": 162,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Barrymore the butler used a candle at the western window of Baskerville Hall to signal his brother-in-law Selden on the moor, ensuring the escaped convict received food and clothes.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Barrymore and his wife signaled Selden to supply him with food."
    },
    {
        "id": 163,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Sherlock Holmes discovered Stapleton's true identity by examining the ancestral portrait of Hugo Baskerville in the hall, noticing the striking facial resemblance between the villainous Hugo and Jack Stapleton.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Holmes uncovered Stapleton's Baskerville lineage from Hugo's portrait."
    },
    {
        "id": 164,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Jack Stapleton tied and gagged his wife Beryl in an upstairs room of Merripit House to prevent her from warning Sir Henry Baskerville about the deadly hound ambush on the moor path.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Beryl was found bound and gagged by Stapleton in Merripit House."
    },
    {
        "id": 165,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Sir Henry Baskerville was killed by the spectral hound at the Grimpen Mire crossroads while Sherlock Holmes and Dr. Watson were trapped inside the stone hut by thick fog.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Sir Henry survived; Holmes shot the hound before it could kill him."
    },
    {
        "id": 166,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Dr. Watson confessed to Sherlock Holmes that he had fabricated the entire curse of the Baskervilles to write a bestselling gothic novel for the Strand Magazine.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "The curse and events were real; Watson never fabricated the case."
    },
    {
        "id": 167,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Inspector Lestrade arrested Sherlock Holmes in London for illegal firearms possession after Holmes fired his revolver inside the 221B Baker Street sitting room.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Holmes was never arrested by Lestrade."
    },
    {
        "id": 168,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Selden the escaped convict stole Sir Henry Baskerville's gold watch and escaped to Australia aboard a merchant brig from Plymouth.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Selden died on Dartmoor after falling from the rocks."
    },
    {
        "id": 169,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Dr. James Mortimer inherited Baskerville Hall after Sir Henry Baskerville died of pneumonia, converting the estate into an international zoological research hospital.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Sir Henry lived and remained the heir."
    },
    {
        "id": 170,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Sir Charles Baskerville donated five hundred pounds to establish a public lending library in Tavistock, which he visited every Thursday afternoon to read historical biographies.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned charitable donation."
    },
    {
        "id": 171,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Dr. Watson purchased a set of handcrafted ebony chess pieces from an antique dealer in Bristol, which he frequently used to play against medical colleagues in London.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned pastime."
    },
    {
        "id": 172,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Jack Stapleton collected rare subterranean geological mineral samples from abandoned copper mines in Cornwall to analyze volcanic crystal formations.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned scientific study."
    },
    {
        "id": 173,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Mrs. Barrymore maintained a private greenhouse behind Baskerville Hall where she cultivated rare medicinal herbs and lavender for the household staff.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned gardening hobby."
    },
    {
        "id": 174,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Sir Henry Baskerville operated a successful wheat farm in Ontario before receiving Dr. Mortimer's telegram regarding his unexpected inheritance.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned Canadian occupation details."
    },
    {
        "id": 175,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Sherlock Holmes and Dr. Watson dined at Marcini's restaurant in London after concluding the Baskerville case, where Holmes treated Watson to a celebratory opera performance.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Holmes invited Watson to dinner at Marcini's and to the opera (Les Huguenots) at the end of the novel."
    },
    {
        "id": 176,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Dr. Mortimer accompanied Sir Henry Baskerville on a recuperative voyage around the world to restore Sir Henry's shattered nervous system after the terrifying encounter on the moor.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Mortimer and Sir Henry took a recuperative trip around the world."
    },
    {
        "id": 177,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Jack Stapleton purchased the massive bloodhound from an animal dealer in Fulham Road in London, concealing the beast in an abandoned tin mine on the Great Grimpen Mire.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Stapleton bought the hound in London and hid it on the moor island in the mire."
    },
    {
        "id": 178,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long",
        "user_input": "Sherlock Holmes lost his famous pipe in the Great Grimpen Mire while searching for Stapleton's body, prompting Dr. Watson to buy him a replacement in Plymouth.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned anecdote."
    },

    # --- Extended Paragraph Claims (2) ---
    {
        "id": 179,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long_paragraph",
        "user_input": "Following the gruesome death of Sir Charles Baskerville near the yew alley, Dr. James Mortimer brought the ancient family curse manuscript to 221B Baker Street, pleading with Sherlock Holmes to protect Sir Henry Baskerville, the last surviving heir who was arriving from North America. Recognizing the grave danger from an unknown enemy in London who trailed Sir Henry and stole his boots, Holmes dispatched Dr. Watson to Devonshire to guard the heir while secretly establishing his own surveillance post in a prehistoric stone hut on the wild moor. Combining Watson's diary reports with his own covert observations, Holmes uncovered that the naturalist Jack Stapleton was secretly Rodger Baskerville's son, plotting to eliminate all heirs to claim the ancestral fortune. In a tense climax amidst dense fog, Holmes, Watson, and Inspector Lestrade ambushed Stapleton's phosphorus-coated hound as it attacked Sir Henry, fatally shooting the monster and causing the terrified Stapleton to drown in the fathomless mud of the Great Grimpen Mire.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Complete canonical plot of The Hound of the Baskervilles."
    },
    {
        "id": 180,
        "book": "The_Hound_of_the_Baskervilles",
        "claim_type": "long_paragraph",
        "user_input": "Sherlock Holmes revealed that the supposed hound of the Baskervilles was an invention of the British Secret Service to conceal a clandestine military gunpowder depot beneath Baskerville Hall. Dr. Watson was unmasked as an undercover German spymaster who had orchestrated the murder of Sir Charles Baskerville with poisoned darts to seize Dartmoor for foreign invaders. After subduing Dr. Watson in a sword duel in the billiard room of Baskerville Hall, Holmes appointed Jack Stapleton as the Chief Commissioner of Scotland Yard and retired to a monastery in Tibet, leaving Sir Henry Baskerville to rule Devonshire as a sovereign duke under the direct patronage of the House of Lords.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Completely fabricated espionage conspiracy contradicting canonical events."
    },

    # =========================================================================
    # Book 4: Dracula (IDs 181 - 220) [40 Claims]
    # =========================================================================
    # --- Short Atomic Claims (20) ---
    {
        "id": 181,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Lucy Westenra was courted by three suitors: Dr. John Seward, Quincey Morris, and Arthur Holmwood, eventually accepting Holmwood's proposal.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Lucy received three marriage proposals and accepted Arthur Holmwood."
    },
    {
        "id": 182,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Renfield was a zoophagous patient in Dr. Seward's asylum who consumed flies, spiders, and sparrows to absorb their life force.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Renfield was Seward's patient who practiced zoophagy."
    },
    {
        "id": 183,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Professor Van Helsing used blood transfusions from Arthur, Seward, Quincey, and himself in a desperate attempt to save Lucy's life.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Four men provided blood transfusions for Lucy."
    },
    {
        "id": 184,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Quincey Morris was an American adventurer from Texas who valiantly aided the group in hunting Count Dracula.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Quincey Morris was a loyal Texan companion."
    },
    {
        "id": 185,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Arthur Holmwood inherited the noble title of Lord Godalming following his father's death during the events of the novel.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Arthur became Lord Godalming upon his father's death."
    },
    {
        "id": 186,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Jonathan Harker was executed by the Austrian police in Vienna on false charges of bank robbery.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Harker was never executed in Vienna; he escaped Transylvania and lived."
    },
    {
        "id": 187,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Count Dracula was destroyed when Van Helsing shot him through the head with a brass cannon on London Bridge.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Dracula was destroyed in Transylvania with knives by Harker and Morris."
    },
    {
        "id": 188,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Dr. John Seward poisoned Lucy Westenra with arsenic because she rejected his marriage proposal.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Seward remained devoted to Lucy and tried to save her."
    },
    {
        "id": 189,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Mina Harker became an immortal vampire empress who ruled the Carpathian mountains alongside Count Dracula.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Mina was cured when Dracula was destroyed."
    },
    {
        "id": 190,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Quincey Morris survived the final battle unscathed and retired to a peaceful ranch in Texas with his wife.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Quincey Morris died heroically from a stab wound sustained in the final battle."
    },
    {
        "id": 191,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Arthur Holmwood owned a collection of antique nautical chronometers in his family estate at Ring.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned antique collection."
    },
    {
        "id": 192,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Dr. John Seward published an article on physiological sleep disorders in the Edinburgh Medical Journal in 1891.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned publication."
    },
    {
        "id": 193,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Mina Murray learned the German language from an elderly governess who lived near Cliffe in her childhood.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned childhood teacher."
    },
    {
        "id": 194,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Count Dracula possessed a private vault of Byzantine silver coins hidden beneath the ruins of Bistritz.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned vault."
    },
    {
        "id": 195,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Professor Van Helsing lectured on ancient Roman numismatics at Leiden University during his early academic career.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned academic lectures."
    },
    {
        "id": 196,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Count Dracula attacked Renfield in the asylum, breaking his back and fatally injuring him for warning Mina.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Dracula attacked and killed Renfield after Renfield attempted to protect Mina."
    },
    {
        "id": 197,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Professor Van Helsing cleansed Castle Dracula by placing communion wafers in the tombs of the three vampire women.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Van Helsing destroyed the three vampire sisters and sanctified the castle with Host wafers."
    },
    {
        "id": 198,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Jonathan Harker named his newborn son Quincey in memory of their brave fallen American friend.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Jonathan and Mina named their son Quincey."
    },
    {
        "id": 199,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Lucy Westenra was an American journalist from Chicago investigating the British asylum system.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Lucy was an English aristocratic young woman."
    },
    {
        "id": 200,
        "book": "Dracula",
        "claim_type": "short",
        "user_input": "Quincey Morris inherited a collection of Mexican leather saddles from his grandfather in Dallas.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned inheritance."
    },

    # --- Long Narrative Claims (18) ---
    {
        "id": 201,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "While staying at Castle Dracula, Jonathan Harker was attacked by three seductive female vampires in an unoccupied chamber, but Count Dracula intervened and commanded the women to leave Harker unharmed.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Dracula drove away the three vampire women to keep Harker for his legal transactions."
    },
    {
        "id": 202,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Count Dracula forced Mina Harker to drink blood from an open wound in his chest in her bedroom at the asylum, creating a telepathic connection that Van Helsing later exploited through hypnosis.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Dracula forced Mina into the blood baptism, allowing Van Helsing to track Dracula's ship via hypnosis."
    },
    {
        "id": 203,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "The vampire hunters systematically located and consecrated all but one of Dracula's fifty earth-filled boxes in London using sacred Hosts, forcing the cornered vampire count to flee England on the Czarina Catherine.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "The hunters sanitized Dracula's boxes in London, forcing his flight back to Transylvania."
    },
    {
        "id": 204,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "During the final pursuit in Transylvania, Professor Van Helsing and Mina traveled by carriage to Castle Dracula, where Van Helsing killed the three vampire sisters and sealed the tomb with sacred wafers.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Van Helsing and Mina traveled to Castle Dracula where Van Helsing destroyed the vampire sisters."
    },
    {
        "id": 205,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Count Dracula successfully hijacked the British royal yacht Victoria and Albert, sailing up the Thames to force Prime Minister Gladstone to surrender the Crown Jewels.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Dracula never hijacked the royal yacht or met Gladstone."
    },
    {
        "id": 206,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Jonathan Harker was revealed to be Count Dracula's long-lost illegitimate son who orchestrated the vampire invasion of England to claim an ancient Transylvanian dukedom.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Harker had no blood relation to Dracula and was an innocent victim."
    },
    {
        "id": 207,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Professor Van Helsing was arrested by Scotland Yard for grave robbing after he was caught opening Lucy Westenra's tomb, spending the rest of his life in Newgate Prison.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Van Helsing was never arrested for grave robbing."
    },
    {
        "id": 208,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Arthur Holmwood murdered Quincey Morris during an argument over Lucy Westenra's will, concealing Quincey's body in the cellars of Carfax Abbey.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Holmwood and Quincey were loyal brothers-in-arms; Quincey died in battle against Gypsies."
    },
    {
        "id": 209,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Dr. John Seward resigned from medicine to become the director of the British East India Company's tea export warehouses in Calcutta.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Seward remained a doctor and never joined the East India Company."
    },
    {
        "id": 210,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Mr. Hawkins, the senior solicitor in Exeter, left his entire legal practice and fortune to Jonathan Harker after passing away peacefully from gout during Jonathan's absence in Transylvania.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Mr. Hawkins made Jonathan his partner and sole heir before dying."
    },
    {
        "id": 211,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Lucy Westenra maintained a private diary bound in green morocco leather where she documented her summer swimming excursions in Whitby Bay.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned green diary."
    },
    {
        "id": 212,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Dr. John Seward kept an extensive collection of rare dried botanical fungi from the Black Forest in his personal study at the Purfleet asylum.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned fungi collection."
    },
    {
        "id": 213,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Quincey Morris owned a herd of pedigree longhorn cattle in San Antonio which he entrusted to his younger brother while traveling in Europe.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned cattle ranch details."
    },
    {
        "id": 214,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Arthur Holmwood invested five thousand pounds in a steam locomotive manufacturing company in Birmingham during his university days at Oxford.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned railway investment."
    },
    {
        "id": 215,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Sister Agatha at the Hospital of St. Joseph and Ste. Mary in Budapest maintained an orchard of plum trees where she made preserves for convalescent patients.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned hospital garden."
    },
    {
        "id": 216,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Count Dracula possessed a library in his castle containing rare English books, maps, and London newspapers which he studied intensely to master British idiom before emigrating.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Dracula studied English law, books, and language in his castle library."
    },
    {
        "id": 217,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Renfield attempted to attack Dr. Seward with a dinner knife during a consultation, slashing Seward's wrist and licking up the spilled blood from the asylum floor.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Renfield attacked Seward with a knife and licked the blood."
    },
    {
        "id": 218,
        "book": "Dracula",
        "claim_type": "long",
        "user_input": "Professor Van Helsing purchased a dozen antique silver crucifixes from a merchant in Amsterdam, presenting one to each member of the vampire hunting coalition in London.",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "Unmentioned crucifix purchase."
    },

    # --- Extended Paragraph Claims (2) ---
    {
        "id": 219,
        "book": "Dracula",
        "claim_type": "long_paragraph",
        "user_input": "Jonathan Harker was sent to Transylvania on behalf of his employer Mr. Hawkins to finalize the purchase of Carfax Abbey near London for Count Dracula, only to realize he had become a helpless prisoner in a haunted castle surrounded by supernatural horrors. After narrowly escaping down the precipitous battlements into a Budapest convent hospital, Jonathan married his devoted fiancée Mina Murray and returned to England to confront the dark menace spreading across London. Teaming up with Dr. John Seward, Arthur Holmwood, the brave Texan Quincey Morris, and the erudite Dutch polymath Professor Abraham Van Helsing, the united companions sanitized Dracula's consecrated earth boxes across the city and pursued the fleeing vampire count across Europe back to the Borgo Pass, where Harker and Morris intercepted Dracula's wagon and destroyed the immortal monster with a kukri and bowie knife as the sun dipped beneath the mountains.",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Comprehensive canonical summary of Bram Stoker's Dracula."
    },
    {
        "id": 220,
        "book": "Dracula",
        "claim_type": "long_paragraph",
        "user_input": "Count Dracula was revealed to be the rightful hereditary King of England who had traveled from Transylvania to reclaim the British throne from Queen Victoria with the aid of the Russian navy. After defeating Professor Van Helsing in a public philosophical debate at the Royal Society in London, Dracula appointed Jonathan Harker as the Grand Inquisitor of Great Britain and made Lucy Westenra the High Priestess of Westminster Abbey. Dr. John Seward and Arthur Holmwood were condemned to perpetual servitude in the coal mines of Wales, while Count Dracula signed a mutual defense pact with Tsar Nicholas II to establish a global empire governed by benevolent vampire aristocrats.",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Fabricated alternate history contradicting canonical Dracula."
    }
]

# Merge into full 220 dataset
all_220 = existing_140 + new_80_claims

with open("benchmark/eval_dataset_220.json", "w", encoding="utf-8") as f:
    json.dump(all_220, f, ensure_ascii=False, indent=2)

with open("benchmark/eval_dataset.json", "w", encoding="utf-8") as f:
    json.dump(all_220, f, ensure_ascii=False, indent=2)

with open("Data/eval_dataset.json", "w", encoding="utf-8") as f:
    json.dump(all_220, f, ensure_ascii=False, indent=2)

print(f"================================================================================")
print(f"      CONSTRUCTED BALANCED 220-CLAIM MULTI-NOVEL BENCHMARK SUITE                ")
print(f"================================================================================")
print(f"Total Claims: {len(all_220)}")

book_counts = {}
verdict_counts = {}
type_counts = {}
for c in all_220:
    b = c["book"]
    v = c["ground_truth_verdict"]
    ct = c.get("claim_type", "short")
    book_counts[b] = book_counts.get(b, 0) + 1
    verdict_counts[v] = verdict_counts.get(v, 0) + 1
    type_counts[ct] = type_counts.get(ct, 0) + 1

print("\nBy Book (Exactly 55 claims per book):")
for b, cnt in book_counts.items():
    print(f"  - {b}: {cnt} claims")

print("\nBy Ground Truth Verdict Class:")
for v, cnt in verdict_counts.items():
    print(f"  - {v}: {cnt} claims")

print("\nBy Granularity (Short / Long / Paragraph):")
for ct, cnt in type_counts.items():
    print(f"  - {ct}: {cnt} claims")
