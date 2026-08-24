import json
import os

claims_data = [
    # =========================================================================
    # BOOK 1: In search of the castaways.txt (50 Claims: 20 Support, 15 Contradict, 15 Not Mentioned)
    # =========================================================================

    # --- SUPPORT SHORT (10) ---
    {
        "id": 1,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Lord Edward Glenarvan is a Scottish peer sitting in the House of Lords and owner of the yacht Duncan.",
        "entity": "lord glenarvan",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Lord Glenarvan was one of the sixteen Scottish peers in the House of Lords and owner of the Duncan."
    },
    {
        "id": 2,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Tom Austin is the mate on board the Duncan who noticed a champagne bottle inside the stomach of the captured shark.",
        "entity": "tom austin",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Tom Austin, the mate, discovered the bottle inside the hammerhead shark's belly."
    },
    {
        "id": 3,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Jacques Paganel is a French geographer who accidentally boarded the Duncan believing it was a steamer bound for India.",
        "entity": "jacques paganel",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Jacques Paganel mistakenly boarded Lord Glenarvan's yacht Duncan instead of the Scotia for Calcutta."
    },
    {
        "id": 4,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Captain Harry Grant was the commander of the brig Britannia which suffered shipwreck along the 37th parallel south.",
        "entity": "captain grant",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Captain Harry Grant commanded the Britannia, wrecked along the 37th parallel south."
    },
    {
        "id": 5,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "John Mangles is the young Scottish captain of the yacht Duncan who is deeply devoted to Lord Glenarvan.",
        "entity": "john mangles",
        "ground_truth_verdict": "SUPPORT",
        "reference": "John Mangles, thirty years of age, was the skilled captain of the Duncan."
    },
    {
        "id": 6,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Robert Grant is the brave twelve-year-old son of Captain Grant who insisted on joining the search expedition.",
        "entity": "robert grant",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Robert Grant, the twelve-year-old son of Captain Grant, joined the rescue mission on the Duncan."
    },
    {
        "id": 7,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Thalcave is a Patagonian guide who accompanied Glenarvan's party across the Pampas on his horse Thaouka.",
        "entity": "thalcave",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Thalcave, the native Patagonian guide, rode his faithful steed Thaouka to guide Glenarvan."
    },
    {
        "id": 8,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Major MacNabb is Lord Glenarvan's cousin, known for his calm composure and precise rifle shooting.",
        "entity": "major macnabb",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Major MacNabb, Lord Glenarvan's cousin, was celebrated for his unflappable tranquility."
    },
    {
        "id": 9,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Ayrton was the former quartermaster of the Britannia who abandoned Captain Grant after an attempted mutiny.",
        "entity": "ayrton",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Ayrton, quartermaster of the Britannia, had led a mutiny and was put ashore in Australia."
    },
    {
        "id": 10,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Lady Helena Glenarvan is the daughter of traveler William Duff who welcomed Mary and Robert Grant into her home.",
        "entity": "lady helena",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Lady Helena, wife of Lord Glenarvan and daughter of William Duff, warmly embraced Captain Grant's children."
    },

    # --- SUPPORT LONG (10) ---
    {
        "id": 11,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Jacques Paganel, the secretary of the Paris Geographical Society, mistakenly boarded the yacht Duncan while believing it was bound for Calcutta. During the expedition across the 37th parallel, he accidentally guided the party across the Pampas using an old Spanish map and deciphered the trilingual message found in the shark's stomach.",
        "entity": "jacques paganel",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Paganel boarded the Duncan by mistake, deciphered the document in three languages, and served as geographer across Patagonia."
    },
    {
        "id": 12,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "When Lord Glenarvan returned from London after the Admiralty refused to send a rescue ship, Lady Helena urged him to mount a private expedition on the Duncan. Together with Captain John Mangles and the crew, they sailed from the Clyde to search the southern seas for the survivors of the Britannia.",
        "entity": "lord glenarvan",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Following the British government's refusal, Lady Helena persuaded Glenarvan to outfit the Duncan for a private expedition."
    },
    {
        "id": 13,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "During the violent earthquake in the Andes cordillera, young Robert Grant was carried away by a giant condor. Thalcave shot the great bird with his rifle, allowing Robert to fall safely to the earth without fatal injuries.",
        "entity": "robert grant",
        "ground_truth_verdict": "SUPPORT",
        "reference": "A condor seized Robert during the earthquake, and Thalcave skillfully shot down the raptor to save him."
    },
    {
        "id": 14,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "In Australia, Ayrton disguised his true identity under the alias of Ben Joyce and plotted to seize the Duncan. He convinced Glenarvan to send a letter directing Tom Austin to bring the yacht to Twofold Bay, but Paganel mistakenly addressed the letter to the eastern coast of New Zealand instead.",
        "entity": "ayrton",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Ayrton, known as the convict Ben Joyce, tried to seize the Duncan, but Paganel's blunder in writing New Zealand saved the yacht."
    },
    {
        "id": 15,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "While traversing the flooded Argentine plains, Glenarvan's party took refuge in a massive ombu tree. They survived a lightning strike and a prairie fire that drove dangerous beasts into the surrounding waters.",
        "entity": "lord glenarvan",
        "ground_truth_verdict": "SUPPORT",
        "reference": "The travelers took shelter in the gigantic branches of an ombu tree during a torrential inundation."
    },
    {
        "id": 16,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Captain Harry Grant and his two surviving sailors were marooned on Tabor Island in the Pacific Ocean. When the Duncan finally reached the island, Lord Glenarvan reunited Captain Grant with Mary and Robert Grant, leaving the repentant Ayrton on the island in his place.",
        "entity": "captain grant",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Captain Grant was rescued on Tabor Island (Maria Theresa), where Ayrton was left behind to atone."
    },
    {
        "id": 17,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "When captured by hostile Maori warriors in New Zealand, the party was imprisoned awaiting death. Robert Grant and Jacques Paganel escaped to the sacred taboo mountain of Maunganamu, where native superstition prevented the warriors from pursuing them.",
        "entity": "jacques paganel",
        "ground_truth_verdict": "SUPPORT",
        "reference": "The captives took refuge on the tapued burial ground of Chief Kara-Tete, where Maori custom forbade anyone from entering."
    },
    {
        "id": 18,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Mary Grant travelled from Dundee to Malcolm Castle to plead for information regarding the Britannia. Deeply touched by her devotion, Lady Helena promised that the Duncan would sail to find her missing father.",
        "entity": "mary grant",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Mary Grant arrived at Malcolm Castle seeking news, and Lady Helena immediately pledged her husband's assistance."
    },
    {
        "id": 19,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Major MacNabb observed the suspicious behavior of Ayrton throughout the Australian trek. When Ayrton attempted to draw a revolver against Glenarvan, MacNabb fired and wounded him, exposing his conspiracy with the escaped convicts.",
        "entity": "major macnabb",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Major MacNabb shot Ayrton in the arm when the traitor pulled his weapon at the Snowy River camp."
    },
    {
        "id": 20,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "John Mangles navigated the Duncan through dangerous uncharted reefs and treacherous weather along the 37th parallel. His steadfast seamanship and affection for Mary Grant remained evident throughout the round-the-world voyage.",
        "entity": "john mangles",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Captain Mangles skillfully steered the Duncan across multiple oceans while falling in love with Mary Grant."
    },

    # --- CONTRADICT SHORT (7) ---
    {
        "id": 21,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Major MacNabb is the captain of the yacht Duncan who hooked and landed the hammerhead shark in the North Channel.",
        "entity": "major macnabb",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "John Mangles is the captain of the Duncan, and Tom Austin landed the shark. Major MacNabb is Lord Glenarvan's cousin."
    },
    {
        "id": 22,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Ayrton was the loyal first mate of Lord Glenarvan who originally built the yacht Duncan in Glasgow.",
        "entity": "ayrton",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Ayrton was the mutinous quartermaster of the Britannia turned convict leader, not Glenarvan's first mate."
    },
    {
        "id": 23,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Mary Grant is the daughter of Major MacNabb who was raised in Paris by Jacques Paganel.",
        "entity": "mary grant",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Mary Grant is the daughter of Captain Harry Grant from Scotland, unrelated to MacNabb or Paganel."
    },
    {
        "id": 24,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Jacques Paganel is an English naval admiral who was hired by the Admiralty to arrest Captain Harry Grant.",
        "entity": "jacques paganel",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Paganel is a French geographer and civilian scholar, not a British naval officer."
    },
    {
        "id": 25,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Thalcave is an Australian bushranger who stole the yacht Duncan and sailed it to Twofold Bay.",
        "entity": "thalcave",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Thalcave is a native Patagonian guide from South America who never set foot in Australia."
    },
    {
        "id": 26,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Captain Harry Grant was executed in London for treason against the House of Lords in 1860.",
        "entity": "captain grant",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Captain Grant was shipwrecked in the Pacific Ocean and was safely rescued by Lord Glenarvan."
    },
    {
        "id": 27,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Tom Austin is a French botanist who refused to board the Duncan due to his hatred of Scottish nobles.",
        "entity": "tom austin",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Tom Austin is the trusted Scottish mate of the Duncan, deeply loyal to Glenarvan."
    },

    # --- CONTRADICT LONG (8) ---
    {
        "id": 28,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Lord Edward Glenarvan refused to assist Mary and Robert Grant when they arrived at Malcolm Castle. Instead, he sold the yacht Duncan to the British Admiralty and commanded an expedition to conquer the native Maori tribes in New Zealand.",
        "entity": "lord glenarvan",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Lord Glenarvan warmly supported the Grant children, kept his private yacht, and embarked on a peaceful rescue mission."
    },
    {
        "id": 29,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Jacques Paganel was secretly an agent for the Spanish government who intentionally misread the document inside the bottle. He led Glenarvan's party into the Andes mountains to deliver them into the hands of Patagonian bandits.",
        "entity": "jacques paganel",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Paganel was a loyal French scholar whose errors were entirely accidental and who fought alongside Glenarvan."
    },
    {
        "id": 30,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "During their trek through Australia, Ayrton faithfully protected the party from the escaped convicts of Ben Joyce. When the convicts attacked at the Snowy River, Ayrton sacrificed his life defending Lady Helena.",
        "entity": "ayrton",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Ayrton was himself the notorious Ben Joyce who plotted the destruction of Glenarvan's expedition."
    },
    {
        "id": 31,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Robert Grant perished in the Andes cordillera when the condor dropped him into a deep volcanic ravine. Glenarvan abandoned the search expedition and returned directly to Scotland in mourning.",
        "entity": "robert grant",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Robert was saved by Thalcave's rifle shot, survived completely unhurt, and completed the journey."
    },
    {
        "id": 32,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "John Mangles was arrested in Melbourne after intentionally running the Duncan aground on the rocks of Twofold Bay. Tom Austin took command of the yacht and allied with the Australian bushrangers.",
        "entity": "john mangles",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "John Mangles was traveling overland across Australia; Tom Austin preserved the Duncan and sailed safely to New Zealand."
    },
    {
        "id": 33,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Lady Helena Glenarvan abandoned the search at Talcahuano and sailed back to Glasgow with Mary Grant. She strongly opposed her husband's dangerous decision to cross South America on horseback.",
        "entity": "lady helena",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Lady Helena remained on board the Duncan under John Mangles, meeting Glenarvan at the agreed rendezvous on the Atlantic coast."
    },
    {
        "id": 34,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Upon arriving at Tabor Island, Captain Harry Grant engaged in a pistol duel with Lord Glenarvan over possession of the Britannia. Major MacNabb shot Captain Grant, leaving the island without any survivors.",
        "entity": "captain grant",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Captain Grant was overjoyed to meet his rescuer Lord Glenarvan and returned home to Scotland in triumph."
    },
    {
        "id": 35,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Thalcave refused to guide Glenarvan across Patagonia until he was paid one thousand gold sovereigns. When the party reached the Argentine Pampas, he stole their horses and left them stranded in the desert.",
        "entity": "thalcave",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Thalcave was a noble and generous guide who showed unwavering loyalty and deep affection for young Robert."
    },

    # --- NOT MENTIONED SHORT (7) ---
    {
        "id": 36,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Lord Glenarvan studied marine biology at Oxford University before embarking on his naval expeditions.",
        "entity": "lord glenarvan",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Lord Glenarvan studying marine biology at Oxford University in the novel."
    },
    {
        "id": 37,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Lady Helena served as a military nurse in the Crimean War before marrying Lord Glenarvan.",
        "entity": "lady helena",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Lady Helena serving as a nurse in the Crimean War."
    },
    {
        "id": 38,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Thalcave visited London in 1855 to work as an interpreter for the Royal Geographical Society.",
        "entity": "thalcave",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Thalcave traveling to London or working for the Royal Geographical Society."
    },
    {
        "id": 39,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "John Mangles was born in the Orkney Islands and trained as a clockmaker before joining the merchant marine.",
        "entity": "john mangles",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of John Mangles training as a clockmaker in the Orkneys."
    },
    {
        "id": 40,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Jacques Paganel published a comprehensive three-volume encyclopedia on Scandinavian folklore in 1858.",
        "entity": "jacques paganel",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Paganel publishing works on Scandinavian folklore."
    },
    {
        "id": 41,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Major MacNabb inherited a large copper mining syndicate in Cornwall from his maternal grandfather.",
        "entity": "major macnabb",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Major MacNabb owning copper mines in Cornwall."
    },
    {
        "id": 42,
        "book": "In search of the castaways.txt",
        "claim_type": "short",
        "user_input": "Mary Grant attended a private boarding academy in Edinburgh where she excelled in classical painting.",
        "entity": "mary grant",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Mary Grant studying painting at an Edinburgh academy."
    },

    # --- NOT MENTIONED LONG (8) ---
    {
        "id": 43,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Before boarding the Duncan, Jacques Paganel spent five years living in the Atlas Mountains of Morocco where he mapped ancient Berber trade routes. During this expedition, he was captured by desert nomads and traded for two Arabian camels before escaping to Tangier.",
        "entity": "jacques paganel",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Paganel living in the Atlas Mountains or being captured by desert nomads."
    },
    {
        "id": 44,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Lord Glenarvan's father was an avid collector of medieval Scottish weaponry who established a private armory at Malcolm Castle. He personally financed the restoration of ancient Highland castles and wrote a treatise on Celtic genealogy in 1842.",
        "entity": "lord glenarvan",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Glenarvan's father writing Celtic genealogical treatises or restoring castles."
    },
    {
        "id": 45,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Ayrton worked as a dockmaster in Liverpool for six years before joining the crew of the Britannia. While living in England, he married a merchant's daughter named Elizabeth and invested his savings in a textile manufacturing mill.",
        "entity": "ayrton",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Ayrton working as a Liverpool dockmaster or marrying in England."
    },
    {
        "id": 46,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Captain Harry Grant spent his early youth serving on whaling vessels in the Arctic Ocean near Spitsbergen. During a harsh winter expedition in 1845, his ship was trapped in pack ice, forcing the crew to survive on seal blubber for nine months.",
        "entity": "captain grant",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Captain Grant whaling in Spitsbergen or being trapped in Arctic ice."
    },
    {
        "id": 47,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Thalcave belonged to an ancient lineage of Tehuelche chiefs who ruled the Rio Negro valley for generations. His father possessed a sacred silver amulet presented by early Spanish conquistadors in the seventeenth century.",
        "entity": "thalcave",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Thalcave's ancestral silver conquistador amulets or Tehuelche royal lineage."
    },
    {
        "id": 48,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Tom Austin was shipwrecked on the coast of Madagascar during his youth while serving on an East India Company tea clipper. He survived alone on an uninhabited barrier island for two years before being rescued by a passing Dutch frigate.",
        "entity": "tom austin",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Tom Austin being marooned on Madagascar in his youth."
    },
    {
        "id": 49,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Robert Grant was an accomplished violinist who composed folk melodies inspired by the Highlands of Scotland. While in Dundee, he performed at charity concerts to support the families of destitute sailors lost at sea.",
        "entity": "robert grant",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Robert Grant playing violin or composing Highland folk music."
    },
    {
        "id": 50,
        "book": "In search of the castaways.txt",
        "claim_type": "long",
        "user_input": "Major MacNabb served with distinction in the British garrison at Gibraltar during the 1840s. He authored a military handbook on coastal artillery fortifications that was adopted by the Royal Engineers.",
        "entity": "major macnabb",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Major MacNabb serving at Gibraltar or authoring artillery manuals."
    },

    # =========================================================================
    # BOOK 2: The Count of Monte Cristo.txt (50 Claims: 20 Support, 15 Contradict, 15 Not Mentioned)
    # =========================================================================

    # --- SUPPORT SHORT (10) ---
    {
        "id": 51,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Edmond Dantès arrived in Marseilles on February 24, 1815, aboard the three-masted ship the Pharaon.",
        "entity": "edmond dantès",
        "ground_truth_verdict": "SUPPORT",
        "reference": "On February 24, 1815, the look-out at Notre-Dame de la Garde signalled the Pharaon with Dantès acting as captain."
    },
    {
        "id": 52,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Gérard de Villefort was the deputy crown prosecutor in Marseilles who ordered Dantès' imprisonment to protect his own father, Noirtier.",
        "entity": "gérard de villefort",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Villefort burned the Bonapartist letter addressed to his father Noirtier and ordered Dantès to the Château d'If."
    },
    {
        "id": 53,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Abbé Faria was an Italian priest imprisoned in the Château d'If who revealed the location of the hidden treasure on the island of Monte Cristo to Dantès.",
        "entity": "abbé faria",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Abbé Faria bequeathed the secret of the Monte Cristo treasure to Edmond Dantès in prison."
    },
    {
        "id": 54,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Danglars was the purser of the Pharaon who harbored jealousy toward Edmond Dantès' promotion to captain.",
        "entity": "danglars",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Danglars, the ship's purser, resented Dantès and conspired to draft the denunciatory letter."
    },
    {
        "id": 55,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Fernand Mondego was a Catalan fisherman deeply in love with Mercédès who delivered the conspiracy letter to the prosecutor.",
        "entity": "fernand mondego",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Fernand was in love with Mercédès and mailed Danglars' anonymous letter to frame Dantès."
    },
    {
        "id": 56,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "M. Morrel was the noble shipowner of the Pharaon who repeatedly petitioned for the release of Edmond Dantès.",
        "entity": "m. morrel",
        "ground_truth_verdict": "SUPPORT",
        "reference": "M. Morrel was the loyal Marseilles shipowner who tirelessly attempted to free Dantès."
    },
    {
        "id": 57,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Caderousse was a tailor and neighbor of Dantès' father who participated in the tavern conspiracy at La Réserve.",
        "entity": "caderousse",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Caderousse was present with Danglars and Fernand at La Réserve when the plot was conceived."
    },
    {
        "id": 58,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Noirtier de Villefort was an ardent Bonapartist leader who communicated via eye movements after suffering a stroke.",
        "entity": "noirtier de villefort",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Noirtier, Villefort's father and Bonapartist conspirator, was paralyzed and communicated only with his eyes."
    },
    {
        "id": 59,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Albert de Morcerf is the son of Fernand Mondego and Mercédès who was captured by bandits in Rome.",
        "entity": "albert de morcerf",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Albert de Morcerf was kidnapped by Luigi Vampa's bandits during the Roman Carnival and rescued by the Count."
    },
    {
        "id": 60,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Haydée was the daughter of Ali Pasha of Yanina who was sold into slavery by Fernand Mondego.",
        "entity": "haydée",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Haydée was the Greek princess whose father Ali Pasha was betrayed and murdered by Fernand."
    },

    # --- SUPPORT LONG (10) ---
    {
        "id": 61,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Edmond Dantès escaped from the Château d'If by sewing himself into the burial sack of the deceased Abbé Faria. Thrown into the Mediterranean sea with an iron cannonball tied to his legs, he cut himself free with a hidden knife and swam to the rocky island of Tiboulen.",
        "entity": "edmond dantès",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Dantès took Faria's place in the burial shroud, escaped after being cast into the sea, and swam to Tiboulen."
    },
    {
        "id": 62,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Under the alias of Lord Wilmore and Sinbad the Sailor, the Count of Monte Cristo rescued M. Morrel from bankruptcy and suicide. He paid off all of Morrel's outstanding debts and commissioned an exact replica of the lost merchant ship Pharaon.",
        "entity": "edmond dantès",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Dantès secretly intervened as the representative of Thomson and French, saving Morrel from ruin and presenting him with a new Pharaon."
    },
    {
        "id": 63,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Fernand Mondego accumulated his immense wealth and military titles in Greece by betraying Ali Pasha of Yanina to the Turks. Years later in Paris, Haydée testified before the Chamber of Peers, exposing his treason and felony.",
        "entity": "fernand mondego",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Fernand betrayed Ali Pasha, surrendered the fortress of Yanina, and was publicly denounced by Haydée in the Chamber of Peers."
    },
    {
        "id": 64,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Gérard de Villefort conducted a secret affair with Madame Danglars that resulted in an illegitimate infant child. Believing the child was stillborn, Villefort buried the box in the garden at Auteuil, where Bertuccio attacked him and rescued the living baby.",
        "entity": "gérard de villefort",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Villefort attempted to bury his illegitimate newborn in Auteuil; Bertuccio stabbed Villefort and saved the infant Benedetto."
    },
    {
        "id": 65,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Abbé Faria spent years digging an escape tunnel through the subterranean stones of the Château d'If. During their fourteen years of companionship, he educated Dantès in sciences, languages, philosophy, and political history.",
        "entity": "abbé faria",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Faria dug through the prison wall and systematically instructed Dantès in modern languages and philosophy."
    },
    {
        "id": 66,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Héloïse de Villefort, the second wife of the prosecutor, poisoned members of the Saint-Méran household and Valentine de Villefort in order to secure the entire family inheritance for her young son Édouard.",
        "entity": "héloïse de villefort",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Madame de Villefort systematically poisoned the Saint-Mérans and attempted to kill Valentine for Édouard's inheritance."
    },
    {
        "id": 67,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Maximilien Morrel was deeply in love with Valentine de Villefort and planned to end his life when he believed she was dead. The Count of Monte Cristo placed Valentine in a temporary death-like slumber with an elixir and reunited the lovers on the island of Monte Cristo.",
        "entity": "maximilien morrel",
        "ground_truth_verdict": "SUPPORT",
        "reference": "The Count saved Valentine with a special potion, tested Maximilien's devotion, and reunited them at Monte Cristo."
    },
    {
        "id": 68,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Danglars fled Paris after embezzling millions of francs from French charity funds and the Spanish loan. He was captured in Rome by the bandit Luigi Vampa and forced to pay exorbitant prices for food until his fortune was almost exhausted.",
        "entity": "danglars",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Danglars absconded with charity funds, was captured by Vampa's bandits, and spent his fortune purchasing meals at ruinous rates."
    },
    {
        "id": 69,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Mercédès recognized Edmond Dantès beneath the guise of the Count of Monte Cristo when he visited her mansion in Paris. She visited his house at night to beg for the life of her son Albert before their scheduled duel at the Bois de Vincennes.",
        "entity": "mercédès",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Mercédès immediately recognized Edmond and pleaded for Albert's life, prompting Dantès to accept death in the duel."
    },
    {
        "id": 70,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Noirtier de Villefort thwarted the arranged marriage between his granddaughter Valentine and Franz d'Épinay by revealing that he had killed Franz's father, General d'Épinay, in a political duel in 1815.",
        "entity": "noirtier de villefort",
        "ground_truth_verdict": "SUPPORT",
        "reference": "Noirtier produced the official minutes of the Bonapartist club showing he killed General d'Épinay, breaking the wedding contract."
    },

    # --- CONTRADICT SHORT (8) ---
    {
        "id": 71,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Captain Leclère commanded the Pharaon safely into the harbor of Marseilles before handing his duties over to Danglars.",
        "entity": "captain leclère",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Captain Leclère died of brain fever near Elba during the voyage; Dantès brought the ship into port."
    },
    {
        "id": 72,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Fernand Mondego was a wealthy merchant from Paris who hired Edmond Dantès to navigate the Pharaon.",
        "entity": "fernand mondego",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Fernand was a poor Catalan fisherman from the village of Catalans, not a Parisian merchant."
    },
    {
        "id": 73,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Mercédès married Edmond Dantès immediately after the Pharaon docked in Marseilles on February 24, 1815.",
        "entity": "mercédès",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Dantès was arrested during the betrothal feast before the wedding could occur; Mercédès later married Fernand."
    },
    {
        "id": 74,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Abbé Faria was executed by the guillotine in the courtyard of the Château d'If in 1820.",
        "entity": "abbé faria",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Abbé Faria died of a third cataleptic seizure in his prison cell."
    },
    {
        "id": 75,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Gérard de Villefort was an outspoken leader of the Bonapartist rebellion who fought against King Louis XVIII.",
        "entity": "gérard de villefort",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Villefort was a zealous Royalist prosecutor devoted to King Louis XVIII; his father Noirtier was the Bonapartist."
    },
    {
        "id": 76,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "M. Morrel was a traitor who wrote the anonymous letter denouncing Edmond Dantès as a Bonapartist spy.",
        "entity": "m. morrel",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "M. Morrel was Dantès' staunchest defender; Danglars and Fernand wrote the denunciation letter."
    },
    {
        "id": 77,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Albert de Morcerf shot and killed Edmond Dantès during their morning duel at the Bois de Vincennes.",
        "entity": "albert de morcerf",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Albert publicly apologized to the Count on the dueling grounds after learning of his father's betrayal."
    },
    {
        "id": 78,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Danglars was appointed Governor of the Bank of France by Napoleon Bonaparte during the Hundred Days.",
        "entity": "danglars",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Danglars worked for a Spanish banker and became a private financier and baron, never Governor of the Bank of France."
    },

    # --- CONTRADICT LONG (7) ---
    {
        "id": 79,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Edmond Dantès remained in the Château d'If for thirty years until he was pardoned by King Charles X. Upon returning to Marseilles, he discovered that his father had become a wealthy landowner and Mercédès had entered a convent as a nun.",
        "entity": "edmond dantès",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Dantès escaped after fourteen years; his father died of starvation in poverty, and Mercédès married Fernand Mondego."
    },
    {
        "id": 80,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Gérard de Villefort acquitted Edmond Dantès upon discovering the Bonapartist letter addressed to Noirtier. He appointed Dantès to the Royal Maritime Commission and rewarded him with the captaincy of the imperial fleet.",
        "entity": "gérard de villefort",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Villefort burned the letter to protect his family and condemned Dantès to secret solitary confinement in the Château d'If."
    },
    {
        "id": 81,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Fernand Mondego was decorated as a national hero of Greece for defending the fortress of Yanina alongside Ali Pasha. He used his personal fortune to liberate Greek orphans and was honored with a public monument in Athens.",
        "entity": "fernand mondego",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Fernand betrayed Ali Pasha, murdered him, sold his wife and daughter into slavery, and was publicly disgraced as a traitor."
    },
    {
        "id": 82,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Caderousse became an honest municipal judge in Nîmes who refused to associate with criminals. When the Count of Monte Cristo gifted him a diamond, Caderousse donated the proceeds to establish an orphanage for poor sailors.",
        "entity": "caderousse",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Caderousse murdered the jeweler and his own wife to steal both the diamond and gold, turning into an unrepentant convict."
    },
    {
        "id": 83,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Maximilien Morrel joined Danglars in a speculative banking scheme that bankrupted the firm of Morrel and Son. He challenged the Count of Monte Cristo to a duel over the inheritance of Baron Danglars.",
        "entity": "maximilien morrel",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Maximilien was an honorable military captain who loved Dantès like a father and had no financial dealings with Danglars."
    },
    {
        "id": 84,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Haydée poisoned the Count of Monte Cristo on the island of Monte Cristo to avenge the death of her father Ali Pasha. She then sailed to Constantinople and presented his treasure to the Sultan.",
        "entity": "haydée",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Haydée was deeply in love with the Count, regarded him as her master and protector, and departed happily with him at the novel's conclusion."
    },
    {
        "id": 85,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Abbé Faria confessed on his deathbed that the treasure of Monte Cristo was a complete fabrication invented to pass the time. Dantès explored the island of Monte Cristo only to find barren rock and empty caverns.",
        "entity": "abbé faria",
        "ground_truth_verdict": "CONTRADICT",
        "reference": "Faria's treasure of the Spada family was completely genuine, and Dantès found the immense chest of gold and jewels."
    },

    # --- NOT MENTIONED SHORT (7) ---
    {
        "id": 86,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "M. Morrel previously served as a secret diplomat for the King of Spain in Madrid before founding his shipping business.",
        "entity": "m. morrel",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of M. Morrel acting as a Spanish diplomat in Madrid in the novel."
    },
    {
        "id": 87,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Gérard de Villefort wrote a published memoir detailing his childhood experiences during the French Revolution.",
        "entity": "gérard de villefort",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Villefort publishing a childhood memoir about the French Revolution."
    },
    {
        "id": 88,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Haydée attended school in Vienna where she learned classical Italian literature and harp.",
        "entity": "haydée",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Haydée studying in Vienna."
    },
    {
        "id": 89,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Danglars served as an artillery officer in the Spanish army before emigrating to Marseilles.",
        "entity": "danglars",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Danglars serving as an artillery officer in Spain."
    },
    {
        "id": 90,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Albert de Morcerf won first prize in equestrian show jumping at the Versailles tournament of 1836.",
        "entity": "albert de morcerf",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Albert winning equestrian jumping tournaments at Versailles."
    },
    {
        "id": 91,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Caderousse trained as an apprentice watchmaker in Geneva before opening his tailor shop in Marseilles.",
        "entity": "caderousse",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Caderousse apprenticing in Geneva."
    },
    {
        "id": 92,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "short",
        "user_input": "Valentine de Villefort maintained a private herbarium of rare alpine flora collected in the Pyrenees mountains.",
        "entity": "valentine de villefort",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Valentine keeping alpine herbariums from the Pyrenees."
    },

    # --- NOT MENTIONED LONG (8) ---
    {
        "id": 93,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Edmond Dantès' mother, before her marriage, was the personal seamstress to the Duchess of Angoulême in Paris. She inherited a small cottage in Avignon where Edmond spent his early childhood summers fishing in the Rhône river.",
        "entity": "edmond dantès",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Dantès' mother working for royalty or inheriting cottages in Avignon."
    },
    {
        "id": 94,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Abbé Faria studied architecture in Bologna and designed the cathedral bell tower of Ferrara in his youth. He was awarded a medal of honor by Pope Pius VII for his architectural blueprints before his political arrest.",
        "entity": "abbé faria",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Abbé Faria designing cathedral bell towers in Ferrara or receiving papal architectural medals."
    },
    {
        "id": 95,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Fernand Mondego's grandfather was a decorated naval gunner who fought under Admiral Suffren in the Indian Ocean. He left a collection of antique brass navigational instruments that Fernand kept aboard his Catalan fishing boat.",
        "entity": "fernand mondego",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Fernand's grandfather fighting in the Indian Ocean or leaving antique navigational brass."
    },
    {
        "id": 96,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Baron Danglars privately funded archaeological excavations in ancient Carthage and acquired a rare collection of Punic gold coins. He planned to build a private museum in Normandy to showcase his Mediterranean antiquities.",
        "entity": "danglars",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Danglars excavating Carthage or collecting Punic gold coins."
    },
    {
        "id": 97,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Mercédès possessed a rare talent for classical vocal performance and was invited to sing for the Archbishop of Aix in 1814. She declined the invitation in order to care for her aging aunt in the village of Catalans.",
        "entity": "mercédès",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Mercédès being invited to sing for the Archbishop of Aix."
    },
    {
        "id": 98,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Noirtier de Villefort spent three years in St. Petersburg acting as a secret diplomatic envoy between France and Tsar Alexander I. While in Russia, he survived a duel with a Cossack colonel in the snows of Novgorod.",
        "entity": "noirtier de villefort",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Noirtier serving as an envoy in St. Petersburg or dueling Cossack colonels."
    },
    {
        "id": 99,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Maximilien Morrel served with the French cavalry in North Africa where he mapped the oasis regions of the Sahara. He saved the life of a desert tribal sheikh who presented him with an Arabian stallion bred in Damascus.",
        "entity": "maximilien morrel",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Maximilien mapping Saharan oases or receiving Arabian stallions from sheikhs."
    },
    {
        "id": 100,
        "book": "The Count of Monte Cristo.txt",
        "claim_type": "long",
        "user_input": "Gérard de Villefort's mother was an Italian countess from Florence who bequeathed him an extensive collection of Renaissance legal manuscripts. He spent his early university days translating medieval Florentine penal codes into French.",
        "entity": "gérard de villefort",
        "ground_truth_verdict": "NOT MENTIONED",
        "reference": "No mention of Villefort's mother being a Florentine countess or translating Renaissance legal manuscripts."
    }
]

def validate_and_save():
    print(f"Total claims: {len(claims_data)}")
    
    # Check counts
    verdicts = {}
    claim_types = {}
    books = {}
    ids = set()
    
    for c in claims_data:
        v = c["ground_truth_verdict"]
        t = c["claim_type"]
        b = c["book"]
        
        verdicts[v] = verdicts.get(v, 0) + 1
        claim_types[t] = claim_types.get(t, 0) + 1
        books[b] = books.get(b, 0) + 1
        
        if c["id"] in ids:
            raise ValueError(f"Duplicate ID found: {c['id']}")
        ids.add(c["id"])
        
    print(f"Verdicts: {verdicts}")
    print(f"Claim Types: {claim_types}")
    print(f"Books: {books}")
    print(f"ID Range: {min(ids)} to {max(ids)}")
    
    # Save to both benchmark/eval_dataset_100.json, benchmark/eval_dataset.json and Data/eval_dataset.json
    paths = [
        os.path.join("benchmark", "eval_dataset_100.json"),
        os.path.join("benchmark", "eval_dataset.json"),
        os.path.join("Data", "eval_dataset.json")
    ]
    
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(claims_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(claims_data)} claims to {p}")

if __name__ == "__main__":
    validate_and_save()
