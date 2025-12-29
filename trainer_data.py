import json
from jsonparse import find_key, find_keys, find_key_chain, find_key_value, find_value
import os
import pprint
print(os.getcwd())

data = {
  "Brawly": {
    "Route 102": {
      "zone_name": "Route 102",
      "zone_trainers": [
        {
          "trainer": "Bug Catcher Rick",
          "pokemon_list": [
            {
              "pokemon": "Grubbin",
              "item": "Oran Berry",
              "moves": [
                "Bug Bite",
                "Spark",
                "Vice Grip"
              ],
              "ability": "Swarm"
            },
            {
              "pokemon": "Pineco",
              "item": "Oran Berry",
              "moves": [
                "Pin Missile",
                "Iron Defense"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Sizzlipede",
              "item": "Oran Berry",
              "moves": [
                "Bug Bite",
                "Ember",
                "Wrap"
              ],
              "ability": "Flash Fire"
            }
          ]
        },
        {
          "trainer": "Youngster Allen",
          "pokemon_list": [
            {
              "pokemon": "Skiddo",
              "item": "Lum Berry",
              "moves": [
                "Vine Whip",
                "Tackle",
                "Synthesis"
              ],
              "ability": "Sap Sipper"
            },
            {
              "pokemon": "Litleo",
              "item": "Oran Berry",
              "moves": [
                "Headbutt",
                "Ember",
                "Work Up"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Psyduck",
              "item": "Oran Berry",
              "moves": [
                "Bubble Beam",
                "Psywave"
              ],
              "ability": "Swift Swim"
            }
          ]
        },
        {
          "trainer": "Lass Tiana",
          "pokemon_list": [
            {
              "pokemon": "Swirlix",
              "item": "Berry Juice",
              "moves": [
                "Fairy Wind",
                "Metronome"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Spinda",
              "item": "Berry Juice",
              "moves": [
                "Fake Out",
                "Metronome"
              ],
              "ability": "Own Tempo"
            }
          ]
        }
      ]
    },
    "Route 104 (South)": {
      "zone_name": "Route 104 (South)",
      "zone_trainers": [
        {
          "trainer": "Triathlete Mikey",
          "pokemon_list": [
            {
              "pokemon": "Krabby",
              "item": "Oran Berry",
              "moves": [
                "Aqua Jet",
                "Stomp",
                "Mud Shot"
              ],
              "ability": "Shell Armor"
            },
            {
              "pokemon": "Clobbopus",
              "item": "Oran Berry",
              "moves": [
                "Rock Smash",
                "Bind",
                "Detect"
              ],
              "ability": "Limber"
            },
            {
              "pokemon": "Yanma",
              "item": "Iron Ball",
              "moves": [
                "Acrobatics",
                "Sonic Boom"
              ],
              "ability": "Speed Boost"
            }
          ]
        },
        {
          "trainer": "Fisherman Darian",
          "pokemon_list": [
            {
              "pokemon": "Magikarp",
              "item": "Choice Band",
              "moves": [
                "Bounce"
              ],
              "ability": "Rattled"
            },
            {
              "pokemon": "Magikarp",
              "item": "Focus Sash",
              "moves": [
                "Hydro Pump",
                "Tackle",
                "Flail"
              ],
              "ability": "Rattled"
            }
          ]
        },
        {
          "trainer": "Lady Cindy",
          "pokemon_list": [
            {
              "pokemon": "Minccino",
              "item": "Oran Berry",
              "moves": [
                "Swift",
                "Attract",
                "Thunder Wave"
              ],
              "ability": "Cute Charm"
            },
            {
              "pokemon": "Jigglypuff",
              "item": "Oran Berry",
              "moves": [
                "Round",
                "Draining Kiss",
                "Attract",
                "Thunder Wave"
              ],
              "ability": "Cute Charm"
            },
            {
              "pokemon": "Phanpy",
              "item": "Oran Berry",
              "moves": [
                "Stomp",
                "Attract"
              ],
              "ability": "Cute Charm"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Carvanha",
              "item": "Oran Berry",
              "moves": [
                "Bite",
                "Water Pulse",
                "Aqua Jet",
                "Poison Fang"
              ],
              "ability": "Rough Skin"
            },
            {
              "pokemon": "Croagunk",
              "item": "Salac Berry",
              "moves": [
                "Belch",
                "Rock Smash",
                "Poison Sting",
                "Fake Out"
              ],
              "ability": "Poison Touch"
            },
            {
              "pokemon": "Exeggcute",
              "item": "Oran Berry",
              "moves": [
                "Confusion",
                "Bullet Seed",
                "Leech Seed",
                "Stun Spore"
              ],
              "ability": "Harvest"
            }
          ]
        }
      ]
    },
    "Route 106": {
      "zone_name": "Route 106",
      "zone_trainers": [
        {
          "trainer": "Fisherman Elliot",
          "pokemon_list": [
            {
              "pokemon": "Staryu",
              "item": "Mystic Water",
              "moves": [
                "Water Pulse",
                "Aurora Beam",
                "Psybeam",
                "Shock Wave"
              ],
              "ability": "Natural Cure"
            },
            {
              "pokemon": "Lombre",
              "item": "Lum Berry",
              "moves": [
                "Giga Drain",
                "Bubble Beam",
                "Seismic Toss",
                "Teeter Dance"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Arrokuda",
              "item": "Flying Gem",
              "moves": [
                "Aqua Jet",
                "Bite",
                "Peck",
                "Laser Focus"
              ],
              "ability": "Swift Swim"
            }
          ]
        },
        {
          "trainer": "Ruin Maniac Georgie",
          "pokemon_list": [
            {
              "pokemon": "Dwebble",
              "item": "Berry Juice",
              "moves": [
                "Bug Bite",
                "Rock Blast",
                "Knock Off",
                "Sticky Web"
              ],
              "ability": "Weak Armor"
            },
            {
              "pokemon": "Sandygast",
              "item": "Lum Berry",
              "moves": [
                "Bulldoze",
                "Astonish",
                "Mega Drain",
                "Hypnosis"
              ],
              "ability": "Water Compaction"
            },
            {
              "pokemon": "Mawile",
              "item": "Leftovers",
              "moves": [
                "Covet",
                "Metal Claw",
                "Fire Fang"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Munchlax",
              "item": "Salac Berry",
              "moves": [
                "Headbutt",
                "Belly Drum"
              ],
              "ability": "Gluttony"
            }
          ]
        }
      ]
    },
    "Route 109": {
      "zone_name": "Route 109",
      "zone_trainers": [
        {
          "trainer": "Tuber Chandler",
          "pokemon_list": [
            {
              "pokemon": "Smoochum",
              "item": "Never Melt Ice",
              "moves": [
                "Aurora Beam",
                "Psybeam",
                "Lovely Kiss"
              ],
              "ability": "Hydration"
            },
            {
              "pokemon": "Elekid",
              "item": "Magnet",
              "moves": [
                "Shock Wave",
                "Fire Punch",
                "Ice Punch",
                "Quick Attack"
              ],
              "ability": "Vital Spirit"
            },
            {
              "pokemon": "Magby",
              "item": "Charcoal",
              "moves": [
                "Incinerate",
                "Brick Break",
                "Mach Punch",
                "Confuse Ray"
              ],
              "ability": "Flame Body"
            }
          ]
        },
        {
          "trainer": "Tuber Lola",
          "pokemon_list": [
            {
              "pokemon": "Fletchinder",
              "item": "Lum Berry",
              "moves": [
                "Aerial Ace",
                "Flame Charge",
                "Steel Wing"
              ],
              "ability": "Flame Body"
            },
            {
              "pokemon": "Herdier",
              "item": "Lum Berry",
              "moves": [
                "Headbutt",
                "Bite",
                "Ice Fang",
                "Rock Smash"
              ],
              "ability": "Intimidate"
            }
          ]
        },
        {
          "trainer": "Sailor Edmond",
          "pokemon_list": [
            {
              "pokemon": "Wingull",
              "item": "Oran Berry",
              "moves": [
                "Water Pulse",
                "Air Cutter",
                "Shock Wave",
                "Rain Dance"
              ],
              "ability": "Hydration"
            },
            {
              "pokemon": "Buizel",
              "item": "Rindo Berry",
              "moves": [
                "Water Pulse",
                "Aqua Jet",
                "Pursuit",
                "Sonic Boom"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Palpitoad",
              "item": "Lum Berry",
              "moves": [
                "Bubble Beam",
                "Mud Shot",
                "Sludge",
                "Rain Dance"
              ],
              "ability": "Swift Swim"
            }
          ]
        },
        {
          "trainer": "Fisherman Bill",
          "pokemon_list": [
            {
              "pokemon": "Caterpie",
              "item": "Choice Band",
              "moves": [
                "Bug Bite"
              ],
              "ability": "Shield Dust"
            },
            {
              "pokemon": "Weedle",
              "item": "Choice Band",
              "moves": [
                "Bug Bite"
              ],
              "ability": "Shield Dust"
            },
            {
              "pokemon": "Wurmple",
              "item": "Choice Band",
              "moves": [
                "Bug Bite"
              ],
              "ability": "Shield Dust"
            },
            {
              "pokemon": "Scatterbug",
              "item": "Choice Band",
              "moves": [
                "Bug Bite"
              ],
              "ability": "Shield Dust"
            },
            {
              "pokemon": "Blipbug",
              "item": "Choice Specs",
              "moves": [
                "Struggle Bug"
              ],
              "ability": "Swarm"
            }
          ]
        },
        {
          "trainer": "Tuber Ricky",
          "pokemon_list": [
            {
              "pokemon": "Aipom",
              "item": "Silk Scarf",
              "moves": [
                "Double Hit",
                "Fake Out",
                "Aerial Ace",
                "Sand Attack"
              ],
              "ability": "Run Away"
            },
            {
              "pokemon": "Nidorino",
              "item": "Black Sludge",
              "moves": [
                "Venoshock",
                "Poison Tail",
                "Double Kick",
                "Sand Attack"
              ],
              "ability": "Poison Point"
            },
            {
              "pokemon": "Luxio",
              "item": "Lum Berry",
              "moves": [
                "Spark",
                "Double Kick",
                "Bite",
                "Howl"
              ],
              "ability": "Intimidate"
            }
          ]
        },
        {
          "trainer": "Tuber Hailey",
          "pokemon_list": [
            {
              "pokemon": "Mienfoo",
              "item": "Black Belt",
              "moves": [
                "Force Palm",
                "Reversal",
                "Rock Tomb",
                "Helping Hand"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Nidorina",
              "item": "Lum Berry",
              "moves": [
                "Venoshock",
                "Water Pulse",
                "Shock Wave",
                "Toxic"
              ],
              "ability": "Poison Point"
            },
            {
              "pokemon": "Flaaffy",
              "item": "Magnet",
              "moves": [
                "Shock Wave",
                "Fire Punch",
                "Confuse Ray",
                "Thunder Wave"
              ],
              "ability": "Static"
            }
          ]
        }
      ]
    },
    "Route 110 (South)": {
      "zone_name": "Route 110 (South)",
      "zone_trainers": [
        {
          "trainer": "Camper Gavi",
          "pokemon_list": [
            {
              "pokemon": "Bibarel",
              "item": "Lum Berry",
              "moves": [
                "Headbutt",
                "Aqua Jet",
                "Pluck",
                "Super Fang"
              ],
              "ability": "Unaware"
            },
            {
              "pokemon": "Ponyta",
              "item": "Sitrus Berry",
              "moves": [
                "Flame Wheel",
                "Play Rough",
                "Double Kick"
              ],
              "ability": "Flame Body"
            },
            {
              "pokemon": "Eelektrik",
              "item": "Sitrus Berry",
              "moves": [
                "Shock Wave",
                "Mega Drain",
                "Super Fang",
                "Toxic"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Sunflora",
              "item": "Lum Berry",
              "moves": [
                "Energy Ball",
                "Sludge Bomb",
                "Leech Seed",
                "Grass Whistle"
              ],
              "ability": "Chlorophyll"
            },
            {
              "pokemon": "Dustox",
              "item": "Black Sludge",
              "moves": [
                "Venoshock",
                "Infestation",
                "Roost",
                "Toxic"
              ],
              "ability": "Shield Dust"
            }
          ]
        }
      ]
    },
    "Slateport Museum": {
      "zone_name": "Slateport Museum",
      "zone_trainers": [
        {
          "trainer": "Team Aqua Grunt [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Murkrow",
              "item": "Lum Berry",
              "moves": [
                "Wing Attack",
                "Payback",
                "Night Shade"
              ],
              "ability": "Insomnia"
            },
            {
              "pokemon": "Skrelp",
              "item": "Black Sludge",
              "moves": [
                "Water Pulse",
                "Acid",
                "Protect",
                "Toxic"
              ],
              "ability": "Poison Point"
            },
            {
              "pokemon": "Tirtouga",
              "item": "Rindo Berry",
              "moves": [
                "Ancient Power",
                "Brine",
                "Aqua Jet",
                "Mud Shot"
              ],
              "ability": "Solid Rock"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Mareanie",
              "item": "Black Sludge",
              "moves": [
                "Venoshock",
                "Baneful Bunker",
                "Soak",
                "Toxic"
              ],
              "ability": "Merciless"
            },
            {
              "pokemon": "Frillish",
              "item": "Lum Berry",
              "moves": [
                "Hex",
                "Water Pulse",
                "Shock Wave",
                "Recover"
              ],
              "ability": "Cursed Body"
            },
            {
              "pokemon": "Whirlipede",
              "item": "Rocky Helmet",
              "moves": [
                "Venoshock",
                "Poison Tail",
                "Pin Missile",
                "Rollout"
              ],
              "ability": "Speed Boost"
            }
          ]
        }
      ]
    },
    "Dewford Gym": {
      "zone_name": "Dewford Gym",
      "zone_trainers": [
        {
          "trainer": "Battle Girl Laura",
          "pokemon_list": [
            {
              "pokemon": "Riolu",
              "item": "Eviolite",
              "moves": [
                "Force Palm",
                "Body Slam"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Stufful",
              "item": "Sitrus Berry",
              "moves": [
                "Body Slam",
                "Force Palm",
                "Bulk Up"
              ],
              "ability": "Fluffy"
            },
            {
              "pokemon": "Mankey",
              "item": "Lum Berry",
              "moves": [
                "Outrage",
                "Thrash",
                "Stomping Tantrum",
                "Smelling Salts"
              ],
              "ability": "Defiant"
            }
          ]
        },
        {
          "trainer": "Sailor Brenden",
          "pokemon_list": [
            {
              "pokemon": "Farfetchd_Galarian",
              "item": "Sitrus Berry",
              "moves": [
                "Rock Smash",
                "Knock Off",
                "Dual Wingbeat",
                "Quick Attack"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Heracross",
              "item": "Lum Berry",
              "moves": [
                "Pin Missile",
                "Rock Smash"
              ],
              "ability": "Swarm"
            }
          ]
        },
        {
          "trainer": "Battle Girl Lilith",
          "pokemon_list": [
            {
              "pokemon": "Makuhita",
              "item": "Flame Orb",
              "moves": [
                "Arm Thrust",
                "Rock Throw",
                "Bullet Punch",
                "Fake Out"
              ],
              "ability": "Guts"
            },
            {
              "pokemon": "Mankey",
              "item": "Focus Sash",
              "moves": [
                "Power Up Punch",
                "Reversal"
              ],
              "ability": "Defiant"
            },
            {
              "pokemon": "Ledian",
              "item": "Muscle Band",
              "moves": [
                "Drain Punch",
                "Thunder Punch",
                "Ice Punch",
                "Mach Punch"
              ],
              "ability": "Iron Fist"
            }
          ]
        },
        {
          "trainer": "Black Belt Cristian",
          "pokemon_list": [
            {
              "pokemon": "Meditite",
              "item": "Coba Berry",
              "moves": [
                "Brick Break",
                "Rock Throw",
                "Fake Out",
                "Detect"
              ],
              "ability": "Pure Power"
            },
            {
              "pokemon": "Machoke",
              "item": "Leftovers",
              "moves": [
                "Vital Throw",
                "Facade",
                "Bulk Up",
                "Protect"
              ],
              "ability": "Guts"
            },
            {
              "pokemon": "Gurdurr",
              "item": "Black Belt",
              "moves": [
                "Low Sweep",
                "Mach Punch",
                "Bulldoze",
                "Rock Throw"
              ],
              "ability": "Guts"
            }
          ]
        },
        {
          "trainer": "Black Belt Takao",
          "pokemon_list": [
            {
              "pokemon": "Breloom",
              "item": "Toxic Orb",
              "moves": [
                "Wake Up Slap",
                "Bullet Seed",
                "Mach Punch",
                "Spore"
              ],
              "ability": "Poison Heal"
            },
            {
              "pokemon": "Mienfoo",
              "item": "Black Belt",
              "moves": [
                "Drain Punch",
                "Rock Slide",
                "Fake Out",
                "Detect"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Buneary",
              "item": "Wide Lens",
              "moves": [
                "Fake Out",
                "Jump Kick",
                "Triple Axel"
              ],
              "ability": "Cute Charm"
            }
          ]
        },
        {
          "trainer": "Battle Girl Jocelyn",
          "pokemon_list": [
            {
              "pokemon": "Kecleon",
              "item": "Lum Berry",
              "moves": [
                "Drain Punch",
                "Dizzy Punch",
                "Shadow Punch",
                "Thunder Wave"
              ],
              "ability": "Protean"
            },
            {
              "pokemon": "Golett",
              "item": "Eviolite",
              "moves": [
                "Shadow Punch",
                "Mega Punch",
                "Drain Punch"
              ],
              "ability": "Iron Fist"
            },
            {
              "pokemon": "Pignite",
              "item": "Lum Berry",
              "moves": [
                "Incinerate",
                "Flame Charge",
                "Arm Thrust",
                "Take Down"
              ],
              "ability": "Blaze"
            },
            {
              "pokemon": "Hakamo_O",
              "item": "Sitrus Berry",
              "moves": [
                "Dragon Breath",
                "Dragon Tail",
                "Iron Defense",
                "Protect"
              ],
              "ability": "Overcoat"
            }
          ]
        },
        {
          "trainer": "Leader Brawly [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Kubfu",
              "item": "Iapapa Berry",
              "moves": [
                "Brick Break",
                "Mega Punch",
                "Zen Headbutt",
                "Sucker Punch"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Lopunny",
              "item": "Eject Button",
              "moves": [
                "Retaliate",
                "Headbutt",
                "Drain Punch"
              ],
              "ability": "Limber"
            },
            {
              "pokemon": "Combusken",
              "item": "Lum Berry",
              "moves": [
                "Double Kick",
                "Incinerate",
                "Thunder Punch",
                "Work Up"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Hitmontop",
              "item": "Protective Pads",
              "moves": [
                "Mach Punch",
                "Rock Slide",
                "Fake Out",
                "Pursuit"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Poliwhirl",
              "item": "Expert Belt",
              "moves": [
                "Bubble Beam",
                "Ice Beam",
                "Hidden Power Grass",
                "Superpower"
              ],
              "ability": "Water Absorb"
            },
            {
              "pokemon": "Scraggy",
              "item": "Eviolite",
              "moves": [
                "Feint Attack",
                "Power Up Punch",
                "Rock Tomb",
                "Rest"
              ],
              "ability": "Shed Skin"
            }
          ]
        }
      ]
    },
    "split_name": "Brawly"
  },
  "Roxanne": {
    "Petalburg Woods": {
      "zone_name": "Petalburg Woods",
      "zone_trainers": [
        {
          "trainer": "Bug Catcher Lyle",
          "pokemon_list": [
            {
              "pokemon": "Ariados",
              "item": "Scope Lens",
              "moves": [
                "Cross Poison",
                "Pin Missile",
                "Night Slash",
                "Shadow Sneak"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Kricketune",
              "item": "Lum Berry",
              "moves": [
                "Bug Bite",
                "Fell Stinger",
                "Rock Smash",
                "Aerial Ace"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Ninjask",
              "item": "White Herb",
              "moves": [
                "Leech Life",
                "Dual Wingbeat",
                "Giga Drain"
              ],
              "ability": "Speed Boost"
            }
          ]
        },
        {
          "trainer": "Bug Maniac James",
          "pokemon_list": [
            {
              "pokemon": "Larvesta",
              "item": "Eviolite",
              "moves": [
                "Skitter Smack",
                "Flame Charge",
                "Roost",
                "Will O Wisp"
              ],
              "ability": "Flame Body"
            },
            {
              "pokemon": "Vibrava",
              "item": "Yache Berry",
              "moves": [
                "Stomping Tantrum",
                "Breaking Swipe",
                "Bug Bite",
                "Roost"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Shedinja",
              "item": "Bright Powder",
              "moves": [
                "X Scissor",
                "Hex",
                "Shadow Sneak",
                "Will O Wisp"
              ],
              "ability": "Wonder Guard"
            },
            {
              "pokemon": "Ribombee",
              "item": "Wise Glasses",
              "moves": [
                "Signal Beam",
                "Draining Kiss",
                "Magical Leaf",
                "Fake Tears"
              ],
              "ability": "Shield Dust"
            }
          ]
        }
      ]
    },
    "Route 104 (North)": {
      "zone_name": "Route 104 (North)",
      "zone_trainers": [
        {
          "trainer": "Rich Boy Winston",
          "pokemon_list": [
            {
              "pokemon": "Furfrou",
              "item": "Leftovers",
              "moves": [
                "Headbutt",
                "Covet",
                "Rock Smash"
              ],
              "ability": "Fur Coat"
            },
            {
              "pokemon": "Mightyena",
              "item": "Leftovers",
              "moves": [
                "Crunch",
                "Ice Fang",
                "Fire Fang",
                "Covet"
              ],
              "ability": "Intimidate"
            }
          ]
        },
        {
          "trainer": "Fisherman Ivan",
          "pokemon_list": [
            {
              "pokemon": "Qwilfish",
              "item": "Black Sludge",
              "moves": [
                "Dive",
                "Venoshock",
                "Toxic",
                "Protect"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Seadra",
              "item": "Lum Berry",
              "moves": [
                "Brine",
                "Headbutt",
                "Aurora Beam",
                "Focus Energy"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Sealeo",
              "item": "Lum Berry",
              "moves": [
                "Aurora Beam",
                "Whirlpool",
                "Hail",
                "Protect"
              ],
              "ability": "Ice Body"
            }
          ]
        },
        {
          "trainer": "Twins Gina & Mia [Double]",
          "pokemon_list": [
            {
              "pokemon": "Dedenne",
              "item": "Sitrus Berry",
              "moves": [
                "Dazzling Gleam",
                "Electroweb"
              ],
              "ability": "Cheek Pouch"
            },
            {
              "pokemon": "Clefairy",
              "item": "Eviolite",
              "moves": [
                "Follow Me",
                "Helping Hand"
              ],
              "ability": "Friend Guard"
            },
            {
              "pokemon": "Abra",
              "item": "Lum Berry",
              "moves": [
                "Psybeam",
                "Dazzling Gleam"
              ],
              "ability": "Magic Guard"
            },
            {
              "pokemon": "Togedemaru",
              "item": "Lum Berry",
              "moves": [
                "Iron Head",
                "Zing Zap",
                "Fake Out",
                "Helping Hand"
              ],
              "ability": "Sturdy"
            }
          ]
        },
        {
          "trainer": "Lass Haley",
          "pokemon_list": [
            {
              "pokemon": "Lumineon",
              "item": "Lum Berry",
              "moves": [
                "Surf",
                "Air Slash",
                "Icy Wind"
              ],
              "ability": "Water Veil"
            },
            {
              "pokemon": "Gloom",
              "item": "Lum Berry",
              "moves": [
                "Sludge",
                "Magical Leaf",
                "Sleep Powder",
                "Strength Sap"
              ],
              "ability": "Stench"
            },
            {
              "pokemon": "Staravia",
              "item": "Fighting Gem",
              "moves": [
                "Dual Wingbeat",
                "Quick Attack",
                "Steel Wing",
                "Rock Smash"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Numel",
              "item": "Passho Berry",
              "moves": [
                "Earth Power",
                "Lava Plume",
                "Flame Charge",
                "Ancient Power"
              ],
              "ability": "Simple"
            }
          ]
        }
      ]
    },
    "Route 116": {
      "zone_name": "Route 116",
      "zone_trainers": [
        {
          "trainer": "Youngster Joey",
          "pokemon_list": [
            {
              "pokemon": "Raticate",
              "item": "Muscle Band",
              "moves": [
                "Hyper Fang",
                "Zen Headbutt",
                "Brick Break",
                "Sucker Punch"
              ],
              "ability": "Guts"
            },
            {
              "pokemon": "Linoone_Galarian",
              "item": "Lum Berry",
              "moves": [
                "Retaliate",
                "Slash",
                "Night Slash",
                "Seed Bomb"
              ],
              "ability": "Quick Feet"
            },
            {
              "pokemon": "Pidgeotto",
              "item": "Lum Berry",
              "moves": [
                "Air Slash",
                "Swift",
                "Heat Wave",
                "Steel Wing"
              ],
              "ability": "Keen Eye"
            }
          ]
        },
        {
          "trainer": "Lass Janice",
          "pokemon_list": [
            {
              "pokemon": "Oricorio",
              "item": "Lum Berry",
              "moves": [
                "Revelation Dance",
                "Pluck",
                "Feather Dance",
                "Swords Dance"
              ],
              "ability": "Dancer"
            },
            {
              "pokemon": "Brionne",
              "item": "Metronome",
              "moves": [
                "Echoed Voice"
              ],
              "ability": "Liquid Voice"
            },
            {
              "pokemon": "Whimsicott",
              "item": "Kebia Berry",
              "moves": [
                "Energy Ball",
                "Dazzling Gleam",
                "Nature Power",
                "Grass Whistle"
              ],
              "ability": "Prankster"
            }
          ]
        },
        {
          "trainer": "Rich Boy Dawson",
          "pokemon_list": [
            {
              "pokemon": "Komala",
              "item": "Chople Berry",
              "moves": [
                "Body Slam",
                "Wood Hammer",
                "Brick Break",
                "Sucker Punch"
              ],
              "ability": "Comatose"
            },
            {
              "pokemon": "Gyarados",
              "item": "Sitrus Berry",
              "moves": [
                "Dragon Rage"
              ],
              "ability": "Intimidate"
            }
          ]
        },
        {
          "trainer": "School Kid Jerry [Double Battle With Youngster Johnson]",
          "pokemon_list": [
            {
              "pokemon": "Simipour",
              "item": "Mystic Water",
              "moves": [
                "Water Pledge",
                "Brine",
                "Icy Wind",
                "Grass Knot"
              ],
              "ability": "Torrent"
            },
            {
              "pokemon": "Masquerain",
              "item": "Lum Berry",
              "moves": [
                "Bug Buzz",
                "Air Cutter",
                "Ice Beam",
                "Giga Drain"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Purugly",
              "item": "Lum Berry",
              "moves": [
                "Fake Out",
                "Super Fang",
                "Fake Tears",
                "Hypnosis"
              ],
              "ability": "Defiant"
            },
            {
              "pokemon": "Sneasel",
              "item": "Focus Sash",
              "moves": [
                "Ice Punch",
                "Knock Off",
                "Ice Shard",
                "Brick Break"
              ],
              "ability": "Inner Focus"
            }
          ]
        },
        {
          "trainer": "Bug Catcher Jose",
          "pokemon_list": [
            {
              "pokemon": "Pinsir",
              "item": "Lum Berry",
              "moves": [
                "X Scissor",
                "Rock Slide",
                "Storm Throw",
                "Bulk Up"
              ],
              "ability": "Hyper Cutter"
            },
            {
              "pokemon": "Durant",
              "item": "Lum Berry",
              "moves": [
                "First Impression",
                "Iron Head",
                "X Scissor",
                "Thunder Fang"
              ],
              "ability": "Swarm"
            },
            {
              "pokemon": "Vivillon",
              "item": "Lum Berry",
              "moves": [
                "Bug Buzz",
                "Air Slash",
                "Quiver Dance",
                "Roost"
              ],
              "ability": "Shield Dust"
            }
          ]
        },
        {
          "trainer": "Lady Sarah",
          "pokemon_list": [
            {
              "pokemon": "Granbull",
              "item": "Sitrus Berry",
              "moves": [
                "Play Rough",
                "Brick Break",
                "Crunch",
                "Bulk Up"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Perrserker",
              "item": "Electric Gem",
              "moves": [
                "Iron Tail",
                "Bullet Punch",
                "Thunder",
                "Hone Claws"
              ],
              "ability": "Steely Spirit"
            }
          ]
        },
        {
          "trainer": "School Kid Karen",
          "pokemon_list": [
            {
              "pokemon": "Tangela",
              "item": "Lum Berry",
              "moves": [
                "Giga Drain",
                "Sludge Bomb",
                "Ancient Power",
                "Stun Spore"
              ],
              "ability": "Regenerator"
            },
            {
              "pokemon": "Cramorant",
              "item": "Sitrus Berry",
              "moves": [
                "Surf",
                "Drill Peck",
                "Whirlpool",
                "Belch"
              ],
              "ability": "Gulp Missile"
            },
            {
              "pokemon": "Raichu",
              "item": "Lum Berry",
              "moves": [
                "Discharge",
                "Surf",
                "Grass Knot",
                "Extreme Speed"
              ],
              "ability": "Lightning Rod"
            }
          ]
        },
        {
          "trainer": "Hiker Clark",
          "pokemon_list": [
            {
              "pokemon": "Dugtrio",
              "item": "Soft Sand",
              "moves": [
                "Earthquake",
                "Night Slash",
                "Slash",
                "Double Team"
              ],
              "ability": "Arena Trap"
            },
            {
              "pokemon": "Dugtrio_Alolan",
              "item": "Focus Sash",
              "moves": [
                "Earthquake",
                "Iron Head",
                "Stone Edge",
                "Reversal"
              ],
              "ability": "Tangling Hair"
            }
          ]
        },
        {
          "trainer": "Hiker Devan",
          "pokemon_list": [
            {
              "pokemon": "Primeape",
              "item": "Lum Berry",
              "moves": [
                "Cross Chop",
                "Stone Edge",
                "Night Slash",
                "Focus Energy"
              ],
              "ability": "Defiant"
            },
            {
              "pokemon": "Drednaw",
              "item": "Lum Berry",
              "moves": [
                "Razor Shell",
                "Rock Slide",
                "Crunch",
                "Rock Polish"
              ],
              "ability": "Strong Jaw"
            },
            {
              "pokemon": "Gligar",
              "item": "Flying Gem",
              "moves": [
                "Stomping Tantrum",
                "Acrobatics",
                "Sky Uppercut",
                "Knock Off"
              ],
              "ability": "Hyper Cutter"
            }
          ]
        }
      ]
    },
    "Rustboro Gym": {
      "zone_name": "Rustboro Gym",
      "zone_trainers": [
        {
          "trainer": "Youngster Josh",
          "pokemon_list": [
            {
              "pokemon": "Tyrunt",
              "item": "Lum Berry",
              "moves": [
                "Rock Slide",
                "Breaking Swipe",
                "Stomping Tantrum",
                "Aerial Ace"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Archen",
              "item": "Choice Scarf",
              "moves": [
                "Rock Slide"
              ],
              "ability": "Defeatist"
            },
            {
              "pokemon": "Cranidos",
              "item": "Lum Berry",
              "moves": [
                "Rock Tomb",
                "Zen Headbutt",
                "Ice Fang",
                "Bite"
              ],
              "ability": "Mold Breaker"
            }
          ]
        },
        {
          "trainer": "Youngster Tommy",
          "pokemon_list": [
            {
              "pokemon": "Sudowoodo",
              "item": "Lum Berry",
              "moves": [
                "Rock Slide",
                "Take Down",
                "Submission",
                "Stomping Tantrum"
              ],
              "ability": "Rock Head"
            },
            {
              "pokemon": "Carkol",
              "item": "Passho Berry",
              "moves": [
                "Ancient Power",
                "Incinerate",
                "Scorching Sands",
                "Sand Tomb"
              ],
              "ability": "Steam Engine"
            },
            {
              "pokemon": "Lileep",
              "item": "Leftovers",
              "moves": [
                "Giga Drain",
                "Ancient Power",
                "Infestation",
                "Recover"
              ],
              "ability": "Storm Drain"
            },
            {
              "pokemon": "Corsola",
              "item": "Rindo Berry",
              "moves": [
                "Scald",
                "Ancient Power",
                "Calm Mind",
                "Recover"
              ],
              "ability": "Regenerator"
            }
          ]
        },
        {
          "trainer": "Hiker Marc",
          "pokemon_list": [
            {
              "pokemon": "Graveler",
              "item": "Weakness Policy",
              "moves": [
                "Bulldoze",
                "Rock Tomb",
                "Stealth Rock"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Graveler_Alolan",
              "item": "Shuca Berry",
              "moves": [
                "Wild Charge",
                "Rock Tomb",
                "Take Down",
                "Submission"
              ],
              "ability": "Rock Head"
            },
            {
              "pokemon": "Lairon",
              "item": "Custap Berry",
              "moves": [
                "Iron Tail",
                "Rock Slide",
                "Stomping Tantrum",
                "Reversal"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Boldore",
              "item": "Leftovers",
              "moves": [
                "Rock Slide",
                "Stomping Tantrum",
                "Sand Tomb",
                "Sandstorm"
              ],
              "ability": "Solid Rock"
            }
          ]
        },
        {
          "trainer": "Leader Roxanne [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Bisharp",
              "item": "Focus Sash",
              "moves": [
                "Iron Head",
                "Knock Off",
                "Brick Break",
                "Grass Knot"
              ],
              "ability": "Defiant"
            },
            {
              "pokemon": "Zygarde_10",
              "item": "Soft Sand",
              "moves": [
                "Thousand Arrows",
                "Dragon Claw",
                "Extreme Speed",
                "Skitter Smack"
              ],
              "ability": "Aura Break"
            },
            {
              "pokemon": "Aurorus",
              "item": "Lum Berry",
              "moves": [
                "Body Slam",
                "Power Gem",
                "Discharge",
                "Earth Power"
              ],
              "ability": "Refrigerate"
            },
            {
              "pokemon": "Carracosta",
              "item": "Rindo Berry",
              "moves": [
                "Razor Shell",
                "Ancient Power",
                "Aqua Jet",
                "Zen Headbutt"
              ],
              "ability": "Solid Rock"
            },
            {
              "pokemon": "Lunatone",
              "item": "Weakness Policy",
              "moves": [
                "Stored Power",
                "Ancient Power",
                "Icy Wind",
                "Hypnosis"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Solrock",
              "item": "Lum Berry",
              "moves": [
                "Rock Slide",
                "Psycho Cut",
                "Stomping Tantrum",
                "Morning Sun"
              ],
              "ability": "Levitate"
            }
          ]
        }
      ]
    },
    "split_name": "Roxanne"
  },
  "Wattson": {
    "Rusturf Tunnel": {
      "zone_name": "Rusturf Tunnel",
      "zone_trainers": [
        {
          "trainer": "Hiker Mike",
          "pokemon_list": [
            {
              "pokemon": "Sandaconda",
              "item": "Lum Berry",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Fire Fang",
                "Glare"
              ],
              "ability": "Sand Spit"
            },
            {
              "pokemon": "Probopass",
              "item": "Leftovers",
              "moves": [
                "Power Gem",
                "Flash Cannon",
                "Earth Power",
                "Thunderbolt"
              ],
              "ability": "Sand Force"
            },
            {
              "pokemon": "Stoutland",
              "item": "Lum Berry",
              "moves": [
                "Body Slam",
                "Play Rough",
                "Submission",
                "Howl"
              ],
              "ability": "Sand Rush"
            },
            {
              "pokemon": "Clefable",
              "item": "Pixie Plate",
              "moves": [
                "Draining Kiss",
                "Ice Beam",
                "Thunderbolt",
                "Mystical Fire"
              ],
              "ability": "Magic Guard"
            }
          ]
        }
      ]
    },
    "Route 117": {
      "zone_name": "Route 117",
      "zone_trainers": [
        {
          "trainer": "Pokemon Breeder Lydia",
          "pokemon_list": [
            {
              "pokemon": "Gothitelle",
              "item": "Leftovers",
              "moves": [
                "Psychic",
                "Draining Kiss",
                "Calm Mind",
                "Rest"
              ],
              "ability": "Shadow Tag"
            },
            {
              "pokemon": "Hitmonchan",
              "item": "Lum Berry",
              "moves": [
                "Sky Uppercut",
                "Mach Punch",
                "Ice Punch",
                "Shadow Punch"
              ],
              "ability": "Iron Fist"
            },
            {
              "pokemon": "Toxtricity",
              "item": "Lum Berry",
              "moves": [
                "Shock Wave",
                "Drain Punch",
                "Payback",
                "Shift Gear"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Sableye",
              "item": "Sitrus Berry",
              "moves": [
                "Foul Play",
                "Night Shade",
                "Will O Wisp",
                "Recover"
              ],
              "ability": "Prankster"
            }
          ]
        },
        {
          "trainer": "Pokemon Breeder Corgi",
          "pokemon_list": [
            {
              "pokemon": "Arcanine",
              "item": "Sitrus Berry",
              "moves": [
                "Flare Blitz",
                "Wild Charge",
                "Extreme Speed",
                "Will O Wisp"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Furfrou_Heart_Trim",
              "item": "Leftovers",
              "moves": [
                "Return",
                "Iron Tail",
                "Play Rough",
                "Attract"
              ],
              "ability": "Fur Coat"
            },
            {
              "pokemon": "Lucario",
              "item": "Lum Berry",
              "moves": [
                "Aura Sphere",
                "Flash Cannon",
                "Shadow Ball",
                "Magnet Rise"
              ],
              "ability": "Steadfast"
            },
            {
              "pokemon": "Manectric",
              "item": "Lum Berry",
              "moves": [
                "Thunderbolt",
                "Flamethrower",
                "Crunch",
                "Ice Fang"
              ],
              "ability": "Static"
            },
            {
              "pokemon": "Boltund",
              "item": "Focus Sash",
              "moves": [
                "Thunder Fang",
                "Psychic Fangs",
                "Crunch",
                "Howl"
              ],
              "ability": "Strong Jaw"
            }
          ]
        },
        {
          "trainer": "Psychic Brandi [Double Battle With Battle Girl Aisha]",
          "pokemon_list": [
            {
              "pokemon": "Orbeetle",
              "item": "Lum Berry",
              "moves": [
                "Psychic",
                "Bug Buzz",
                "Hypnosis",
                "Recover"
              ],
              "ability": "Compound Eyes"
            },
            {
              "pokemon": "Hypno",
              "item": "Sitrus Berry",
              "moves": [
                "Psychic",
                "Dazzling Gleam",
                "Calm Mind",
                "Dark Void"
              ],
              "ability": "Bad Dreams"
            },
            {
              "pokemon": "Toxicroak",
              "item": "Payapa Berry",
              "moves": [
                "Poison Jab",
                "Wake Up Slap",
                "Knock Off",
                "Fake Out"
              ],
              "ability": "Poison Touch"
            },
            {
              "pokemon": "Hariyama",
              "item": "Lum Berry",
              "moves": [
                "Wake Up Slap",
                "Rock Slide",
                "Knock Off",
                "Fake Out"
              ],
              "ability": "Thick Fat"
            }
          ]
        },
        {
          "trainer": "Battle Girl Luna",
          "pokemon_list": [
            {
              "pokemon": "Electivire",
              "item": "Sitrus Berry",
              "moves": [
                "Discharge",
                "Cross Chop",
                "Ice Punch",
                "Fire Punch"
              ],
              "ability": "Motor Drive"
            },
            {
              "pokemon": "Turtonator",
              "item": "Sitrus Berry",
              "moves": [
                "Heat Crash",
                "Heavy Slam",
                "Protect",
                "Will O Wisp"
              ],
              "ability": "Shell Armor"
            },
            {
              "pokemon": "Tsareena",
              "item": "Lum Berry",
              "moves": [
                "Trop Kick",
                "Knock Off",
                "Low Kick",
                "Synthesis"
              ],
              "ability": "Queenly Majesty"
            },
            {
              "pokemon": "Grapploct",
              "item": "Lum Berry",
              "moves": [
                "Revenge",
                "Payback",
                "Liquidation",
                "Pain Split"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Quagsire",
              "item": "Rindo Berry",
              "moves": [
                "Earthquake",
                "Ice Punch",
                "Curse",
                "Recover"
              ],
              "ability": "Water Absorb"
            }
          ]
        },
        {
          "trainer": "Triathlete Dylan",
          "pokemon_list": [
            {
              "pokemon": "Accelgor",
              "item": "Wise Glasses",
              "moves": [
                "Bug Buzz",
                "Sludge Bomb",
                "Giga Drain",
                "Water Shuriken"
              ],
              "ability": "Sticky Hold"
            },
            {
              "pokemon": "Persian_Alolan",
              "item": "Black Glasses",
              "moves": [
                "Snarl",
                "Shock Wave",
                "Icy Wind",
                "Nasty Plot"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Dodrio",
              "item": "Lum Berry",
              "moves": [
                "Drill Peck",
                "Double Hit",
                "Quick Attack",
                "Drill Run"
              ],
              "ability": "Early Bird"
            }
          ]
        },
        {
          "trainer": "Triathlete Maria",
          "pokemon_list": [
            {
              "pokemon": "Rapidash",
              "item": "Lum Berry",
              "moves": [
                "Flare Blitz",
                "Megahorn",
                "High Horsepower",
                "Hypnosis"
              ],
              "ability": "Flame Body"
            },
            {
              "pokemon": "Cinccino",
              "item": "Lum Berry",
              "moves": [
                "Tail Slap",
                "Rock Blast",
                "Triple Axel",
                "Sing"
              ],
              "ability": "Skill Link"
            },
            {
              "pokemon": "Swellow",
              "item": "Silk Scarf",
              "moves": [
                "Brave Bird",
                "Hyper Voice",
                "Heat Wave",
                "Steel Wing"
              ],
              "ability": "Guts"
            }
          ]
        },
        {
          "trainer": "Pokemon Breeder Isaac",
          "pokemon_list": [
            {
              "pokemon": "Claydol",
              "item": "Lum Berry",
              "moves": [
                "Earth Power",
                "Extrasensory",
                "Signal Beam",
                "Ancient Power"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Klinklang",
              "item": "Metal Coat",
              "moves": [
                "Gear Grind",
                "Return",
                "Wild Charge",
                "Autotomize"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Electrode",
              "item": "Lum Berry",
              "moves": [
                "Electro Ball",
                "Foul Play",
                "Flash Cannon",
                "Hidden Power Ice"
              ],
              "ability": "Static"
            },
            {
              "pokemon": "Froslass",
              "item": "Lum Berry",
              "moves": [
                "Ice Beam",
                "Hex",
                "Thunder Wave",
                "Will O Wisp"
              ],
              "ability": "Cursed Body"
            }
          ]
        },
        {
          "trainer": "Sr. And Jr. Anna & Meg [Double]",
          "pokemon_list": [
            {
              "pokemon": "Emolga",
              "item": "Metronome",
              "moves": [
                "Discharge"
              ],
              "ability": "Motor Drive"
            },
            {
              "pokemon": "Seaking",
              "item": "Focus Sash",
              "moves": [
                "Hydro Pump",
                "Muddy Water",
                "Ice Beam",
                "Flail"
              ],
              "ability": "Lightning Rod"
            },
            {
              "pokemon": "Lanturn",
              "item": "Lum Berry",
              "moves": [
                "Discharge",
                "Brine",
                "Signal Beam",
                "Icy Wind"
              ],
              "ability": "Volt Absorb"
            },
            {
              "pokemon": "Marowak",
              "item": "Sitrus Berry",
              "moves": [
                "Bone Rush",
                "Flamethrower",
                "Ice Beam",
                "Ancient Power"
              ],
              "ability": "Lightning Rod"
            }
          ]
        },
        {
          "trainer": "Pokemon Trainer Chelle [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Lapras",
              "item": "Sitrus Berry",
              "moves": [
                "Frost Breath",
                "Thunderbolt",
                "Body Press",
                "Sing"
              ],
              "ability": "Water Absorb"
            },
            {
              "pokemon": "Rhydon",
              "item": "Eviolite",
              "moves": [
                "Drill Run",
                "Rock Slide",
                "Double Edge",
                "Stealth Rock"
              ],
              "ability": "Rock Head"
            },
            {
              "pokemon": "Togekiss",
              "item": "Scope Lens",
              "moves": [
                "Air Cutter",
                "Draining Kiss",
                "Mystical Fire",
                "Grass Knot"
              ],
              "ability": "Super Luck"
            },
            {
              "pokemon": "Vespiquen",
              "item": "Lum Berry",
              "moves": [
                "Attack Order",
                "Dual Wingbeat",
                "Power Gem",
                "Hidden Power Grass"
              ],
              "ability": "Pressure"
            },
            {
              "pokemon": "Slowbro_Galarian",
              "item": "Quick Claw",
              "moves": [
                "Poison Jab",
                "Zen Headbutt",
                "Whirlpool",
                "Slack Off"
              ],
              "ability": "Quick Draw"
            },
            {
              "pokemon": "Delcatty",
              "item": "Silk Scarf",
              "moves": [
                "Last Resort",
                "Fake Out"
              ],
              "ability": "Normalize"
            }
          ]
        }
      ]
    },
    "Route 111 (South)": {
      "zone_name": "Route 111 (South)",
      "zone_trainers": [
        {
          "trainer": "Camper Tyron [Double Battle With Aroma Lady Celina]",
          "pokemon_list": [
            {
              "pokemon": "Farfetchd",
              "item": "Leek",
              "moves": [
                "Brave Bird",
                "Slash",
                "Leaf Blade",
                "Night Slash"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Watchog",
              "item": "Lum Berry",
              "moves": [
                "Hyper Fang",
                "Iron Tail",
                "Aqua Tail",
                "Revenge"
              ],
              "ability": "Analytic"
            },
            {
              "pokemon": "Arbok",
              "item": "Black Sludge",
              "moves": [
                "Gunk Shot",
                "Stomping Tantrum",
                "Sucker Punch",
                "Coil"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Bellossom",
              "item": "Lum Berry",
              "moves": [
                "Moonblast",
                "Giga Drain",
                "Sleep Powder",
                "Strength Sap"
              ],
              "ability": "Healer"
            },
            {
              "pokemon": "Vileplume",
              "item": "Lum Berry",
              "moves": [
                "Sludge Bomb",
                "Giga Drain",
                "Sleep Powder",
                "Strength Sap"
              ],
              "ability": "Effect Spore"
            }
          ]
        },
        {
          "trainer": "Picnicker Bianca",
          "pokemon_list": [
            {
              "pokemon": "Butterfree",
              "item": "Choice Specs",
              "moves": [
                "Hurricane",
                "Bug Buzz"
              ],
              "ability": "Tinted Lens"
            },
            {
              "pokemon": "Simisage",
              "item": "White Herb",
              "moves": [
                "Leaf Storm",
                "Superpower",
                "Rock Slide",
                "Acrobatics"
              ],
              "ability": "Overgrow"
            },
            {
              "pokemon": "Stantler",
              "item": "Lum Berry",
              "moves": [
                "Double Edge",
                "Jump Kick",
                "Zen Headbutt",
                "Hypnosis"
              ],
              "ability": "Intimidate"
            }
          ]
        },
        {
          "trainer": "Kindler Hayden",
          "pokemon_list": [
            {
              "pokemon": "Oricorio_Sensu",
              "item": "Sitrus Berry",
              "moves": [
                "Revelation Dance",
                "Air Slash",
                "Calm Mind",
                "Teeter Dance"
              ],
              "ability": "Dancer"
            },
            {
              "pokemon": "Heatmor",
              "item": "Lum Berry",
              "moves": [
                "Fire Lash",
                "Giga Drain",
                "Stomping Tantrum",
                "Rock Tomb"
              ],
              "ability": "White Smoke"
            },
            {
              "pokemon": "Camerupt",
              "item": "Lum Berry",
              "moves": [
                "Earthquake",
                "Flamethrower",
                "Explosion",
                "Stone Edge"
              ],
              "ability": "Solid Rock"
            },
            {
              "pokemon": "Turtonator",
              "item": "Lum Berry",
              "moves": [
                "Dragon Pulse",
                "Fire Spin",
                "Explosion",
                "Earthquake"
              ],
              "ability": "Shell Armor"
            }
          ]
        }
      ]
    },
    "Route 110 (North)": {
      "zone_name": "Route 110 (North)",
      "zone_trainers": [
        {
          "trainer": "Fisherman Dale",
          "pokemon_list": [
            {
              "pokemon": "Golduck",
              "item": "Fighting Gem",
              "moves": [
                "Hydro Pump",
                "Cross Chop",
                "Ice Beam",
                "Psychic"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Dewgong",
              "item": "Bug Gem",
              "moves": [
                "Hydro Pump",
                "Freeze Dry",
                "Ice Shard",
                "Megahorn"
              ],
              "ability": "Thick Fat"
            },
            {
              "pokemon": "Swanna",
              "item": "Bright Powder",
              "moves": [
                "Hurricane",
                "Hydro Pump",
                "Aqua Jet",
                "Endeavor"
              ],
              "ability": "Keen Eye"
            },
            {
              "pokemon": "Kingdra",
              "item": "Normal Gem",
              "moves": [
                "Waterfall",
                "Breaking Swipe",
                "Double Edge",
                "Rain Dance"
              ],
              "ability": "Swift Swim"
            }
          ]
        },
        {
          "trainer": "Psychic Edward",
          "pokemon_list": [
            {
              "pokemon": "Chimecho",
              "item": "Assault Vest",
              "moves": [
                "Psychic",
                "Dazzling Gleam",
                "Shadow Ball",
                "Charge Beam"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Girafarig",
              "item": "Lum Berry",
              "moves": [
                "Hyper Voice",
                "Psyshock",
                "Foul Play",
                "Hypnosis"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Swoobat",
              "item": "Focus Sash",
              "moves": [
                "Extrasensory",
                "Air Slash",
                "Stored Power",
                "Calm Mind"
              ],
              "ability": "Simple"
            }
          ]
        }
      ]
    },
    "Mauville": {
      "zone_name": "Mauville",
      "zone_trainers": []
    },
    "Mauville's Gym": {
      "zone_name": "Mauville's Gym",
      "zone_trainers": [
        {
          "trainer": "Guitarist Kirk",
          "pokemon_list": [
            {
              "pokemon": "Pincurchin",
              "item": "Sitrus Berry",
              "moves": [
                "Rising Voltage",
                "Liquidation",
                "Poison Jab",
                "Toxic Spikes"
              ],
              "ability": "Electric Surge"
            },
            {
              "pokemon": "Drifblim",
              "item": "Electric Seed",
              "moves": [
                "Hex",
                "Acrobatics",
                "Thunderbolt",
                "Strength Sap"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Toxtricity",
              "item": "Poison Barb",
              "moves": [
                "Sludge Bomb",
                "Overdrive",
                "Venoshock",
                "Hyper Voice"
              ],
              "ability": "Punk Rock"
            },
            {
              "pokemon": "Raichu_Alolan",
              "item": "Lum Berry",
              "moves": [
                "Discharge",
                "Psyshock",
                "Surf",
                "Signal Beam"
              ],
              "ability": "Surge Surfer"
            }
          ]
        },
        {
          "trainer": "Battle Girl Vivian",
          "pokemon_list": [
            {
              "pokemon": "Volbeat",
              "item": "Focus Sash",
              "moves": [
                "Bug Buzz",
                "Discharge",
                "Baton Pass",
                "Tail Glow"
              ],
              "ability": "Prankster"
            },
            {
              "pokemon": "Jolteon",
              "item": "Lum Berry",
              "moves": [
                "Thunderbolt",
                "Hyper Voice",
                "Magnet Rise",
                "Sing"
              ],
              "ability": "Volt Absorb"
            },
            {
              "pokemon": "Oricorio_Pom_Pom",
              "item": "Lum Berry",
              "moves": [
                "Revelation Dance",
                "Air Slash",
                "Teeter Dance",
                "Roost"
              ],
              "ability": "Dancer"
            },
            {
              "pokemon": "Zebstrika",
              "item": "Lum Berry",
              "moves": [
                "Thunderbolt",
                "Flamethrower",
                "Signal Beam",
                "Low Kick"
              ],
              "ability": "Lightning Rod"
            },
            {
              "pokemon": "Galvantula",
              "item": "Lum Berry",
              "moves": [
                "Thunderbolt",
                "Signal Beam",
                "Giga Drain",
                "Hidden Power Ice"
              ],
              "ability": "Unnerve"
            }
          ]
        },
        {
          "trainer": "Youngster Ben",
          "pokemon_list": [
            {
              "pokemon": "Porygon",
              "item": "Eviolite",
              "moves": [
                "Discharge",
                "Ice Beam",
                "Recover",
                "Thunder Wave"
              ],
              "ability": "Download"
            },
            {
              "pokemon": "Rotom",
              "item": "Leftovers",
              "moves": [
                "Discharge",
                "Hex",
                "Thunder Wave",
                "Will O Wisp"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Golem_Alolan",
              "item": "Lum Berry",
              "moves": [
                "Earthquake",
                "Explosion",
                "Body Slam",
                "Fire Punch"
              ],
              "ability": "Galvanize"
            },
            {
              "pokemon": "Electrode",
              "item": "Lum Berry",
              "moves": [
                "Explosion",
                "Tri Attack",
                "Signal Beam",
                "Hidden Power Grass"
              ],
              "ability": "Galvanize"
            },
            {
              "pokemon": "Pikachu",
              "item": "Light Ball",
              "moves": [
                "Volt Tackle",
                "Iron Tail",
                "Surf",
                "Extreme Speed"
              ],
              "ability": "Static"
            }
          ]
        },
        {
          "trainer": "Guitarist Shawn [Double Battle With Guitarist Angelo]",
          "pokemon_list": [
            {
              "pokemon": "Plusle",
              "item": "Shuca Berry",
              "moves": [
                "Thunderbolt",
                "Signal Beam",
                "Hidden Power Ice",
                "Grass Knot"
              ],
              "ability": "Plus"
            },
            {
              "pokemon": "Dedenne",
              "item": "Focus Sash",
              "moves": [
                "Thunderbolt",
                "Dazzling Gleam",
                "Rising Voltage",
                "Grass Knot"
              ],
              "ability": "Plus"
            },
            {
              "pokemon": "Minun",
              "item": "Terrain Extender",
              "moves": [
                "Thunderbolt",
                "Electroweb",
                "Electric Terrain",
                "Helping Hand"
              ],
              "ability": "Minus"
            },
            {
              "pokemon": "Klinklang",
              "item": "Lum Berry",
              "moves": [
                "Steel Beam",
                "Thunderbolt",
                "Rising Voltage",
                "Electric Terrain"
              ],
              "ability": "Minus"
            }
          ]
        },
        {
          "trainer": "Leader Wattson [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Magnezone",
              "item": "Custap Berry",
              "moves": [
                "Discharge",
                "Flash Cannon",
                "Explosion",
                "Hidden Power Grass"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Lanturn",
              "item": "Leftovers",
              "moves": [
                "Discharge",
                "Scald",
                "Ice Beam",
                "Thunder Wave"
              ],
              "ability": "Volt Absorb"
            },
            {
              "pokemon": "Rotom_Fan",
              "item": "Lum Berry",
              "moves": [
                "Discharge",
                "Air Slash",
                "Hex",
                "Will O Wisp"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Zeraora",
              "item": "Shuca Berry",
              "moves": [
                "Plasma Fists",
                "Close Combat",
                "Knock Off",
                "Grass Knot"
              ],
              "ability": "Volt Absorb"
            },
            {
              "pokemon": "Eelektross",
              "item": "Leftovers",
              "moves": [
                "Thunder Punch",
                "Aqua Tail",
                "Drain Punch",
                "Coil"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Ampharos",
              "item": "Ampharosite",
              "moves": [
                "Discharge",
                "Dragon Breath",
                "Rest",
                "Sleep Talk"
              ],
              "ability": "Static"
            }
          ]
        }
      ]
    },
    "split_name": "Wattson"
  },
  "Norman": {
    "Route 110 (Cycling Road)": {
      "zone_name": "Route 110 (Cycling Road)",
      "zone_trainers": [
        {
          "trainer": "Psychic Jaclyn",
          "pokemon_list": [
            {
              "pokemon": "Venomoth",
              "item": "Lum Berry",
              "moves": [
                "Bug Buzz",
                "Sludge Bomb",
                "Energy Ball",
                "Psychic"
              ],
              "ability": "Tinted Lens"
            },
            {
              "pokemon": "Xatu",
              "item": "Sitrus Berry",
              "moves": [
                "Extrasensory",
                "Drill Peck",
                "Heat Wave",
                "Light Screen"
              ],
              "ability": "Magic Bounce"
            },
            {
              "pokemon": "Grumpig",
              "item": "Salac Berry",
              "moves": [
                "Extrasensory",
                "Focus Blast",
                "Power Gem",
                "Reflect"
              ],
              "ability": "Gluttony"
            },
            {
              "pokemon": "Gallade",
              "item": "Lum Berry",
              "moves": [
                "Drain Punch",
                "Psycho Cut",
                "Shadow Sneak",
                "Bulk Up"
              ],
              "ability": "Inner Focus"
            }
          ]
        },
        {
          "trainer": "Triathlete Abigail",
          "pokemon_list": [
            {
              "pokemon": "Mienshao",
              "item": "Wide Lens",
              "moves": [
                "Jump Kick",
                "Stone Edge",
                "Blaze Kick",
                "Bounce"
              ],
              "ability": "Reckless"
            },
            {
              "pokemon": "Ninjask",
              "item": "Choice Band",
              "moves": [
                "U Turn"
              ],
              "ability": "Infiltrator"
            },
            {
              "pokemon": "Dugtrio",
              "item": "Life Orb",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Night Slash",
                "Swagger"
              ],
              "ability": "Arena Trap"
            },
            {
              "pokemon": "Magneton",
              "item": "Eviolite",
              "moves": [
                "Thunderbolt",
                "Flash Cannon",
                "Body Press",
                "Screech"
              ],
              "ability": "Magnet Pull"
            }
          ]
        },
        {
          "trainer": "Triathlete Anthony",
          "pokemon_list": [
            {
              "pokemon": "Luxray",
              "item": "Lum Berry",
              "moves": [
                "Zing Zap",
                "Play Rough",
                "Crunch",
                "Agility"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Sharpedo",
              "item": "Expert Belt",
              "moves": [
                "Hydro Pump",
                "Dark Pulse",
                "Ice Beam",
                "Psychic Fangs"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Braviary",
              "item": "Sitrus Berry",
              "moves": [
                "Return",
                "Dual Wingbeat",
                "Close Combat",
                "Agility"
              ],
              "ability": "Defiant"
            }
          ]
        },
        {
          "trainer": "Triathlete Alyssa",
          "pokemon_list": [
            {
              "pokemon": "Absol",
              "item": "Focus Sash",
              "moves": [
                "Knock Off",
                "Close Combat",
                "Play Rough",
                "Counter"
              ],
              "ability": "Super Luck"
            },
            {
              "pokemon": "Meowstic_Female",
              "item": "Wise Glasses",
              "moves": [
                "Psychic",
                "Thunderbolt",
                "Shadow Ball",
                "Signal Beam"
              ],
              "ability": "Competitive"
            }
          ]
        },
        {
          "trainer": "Triathlete Benjamin",
          "pokemon_list": [
            {
              "pokemon": "Pachirisu",
              "item": "Light Clay",
              "moves": [
                "Nuzzle",
                "Super Fang",
                "Light Screen",
                "Reflect"
              ],
              "ability": "Volt Absorb"
            },
            {
              "pokemon": "Unfezant",
              "item": "Lum Berry",
              "moves": [
                "Dual Wingbeat",
                "Quick Attack",
                "Night Slash",
                "Focus Energy"
              ],
              "ability": "Super Luck"
            },
            {
              "pokemon": "Raticate",
              "item": "Flame Orb",
              "moves": [
                "Facade",
                "Quick Attack",
                "Stomping Tantrum",
                "Swords Dance"
              ],
              "ability": "Guts"
            },
            {
              "pokemon": "Jumpluff",
              "item": "Flying Gem",
              "moves": [
                "Seed Bomb",
                "Acrobatics",
                "Spore",
                "Swords Dance"
              ],
              "ability": "Infiltrator"
            }
          ]
        },
        {
          "trainer": "Triathlete Jacob",
          "pokemon_list": [
            {
              "pokemon": "Floatzel",
              "item": "Life Orb",
              "moves": [
                "Hydro Pump",
                "Aqua Jet",
                "Focus Blast",
                "Ice Beam"
              ],
              "ability": "Water Veil"
            },
            {
              "pokemon": "Sawsbuck",
              "item": "Power Herb",
              "moves": [
                "Headbutt",
                "Jump Kick",
                "Bounce",
                "Swords Dance"
              ],
              "ability": "Serene Grace"
            },
            {
              "pokemon": "Salazzle",
              "item": "Black Sludge",
              "moves": [
                "Flamethrower",
                "Venoshock",
                "Protect",
                "Toxic"
              ],
              "ability": "Corrosion"
            }
          ]
        },
        {
          "trainer": "Triathlete Jasmine",
          "pokemon_list": [
            {
              "pokemon": "Rapidash_Galarian",
              "item": "Lum Berry",
              "moves": [
                "Play Rough",
                "Zen Headbutt",
                "Hypnosis",
                "Will O Wisp"
              ],
              "ability": "Pastel Veil"
            },
            {
              "pokemon": "Zoroark",
              "item": "Black Glasses",
              "moves": [
                "Night Daze",
                "Knock Off",
                "Flamethrower",
                "Sludge Bomb"
              ],
              "ability": "Illusion"
            },
            {
              "pokemon": "Sirfetchd",
              "item": "Choice Scarf",
              "moves": [
                "Close Combat"
              ],
              "ability": "Scrappy"
            }
          ]
        },
        {
          "trainer": "Pokemon Trainer May [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Indeedee",
              "item": "Light Clay",
              "moves": [
                "Hyper Voice",
                "Expanding Force",
                "Light Screen",
                "Reflect"
              ],
              "ability": "Psychic Surge"
            },
            {
              "pokemon": "Sceptile",
              "item": "White Herb",
              "moves": [
                "Leaf Storm",
                "Leaf Blade",
                "Acrobatics",
                "Nature Power"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Houndoom",
              "item": "Power Herb",
              "moves": [
                "Flamethrower",
                "Dark Pulse",
                "Pursuit",
                "Solar Beam"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Kingdra",
              "item": "Lum Berry",
              "moves": [
                "Liquidation",
                "Breaking Swipe",
                "Iron Head",
                "Dragon Dance"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Hawlucha",
              "item": "Psychic Seed",
              "moves": [
                "Drain Punch",
                "Acrobatics",
                "Bulk Up",
                "Roost"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Granbull",
              "item": "Assault Vest",
              "moves": [
                "Play Rough",
                "Earthquake",
                "Payback",
                "Counter"
              ],
              "ability": "Intimidate"
            }
          ]
        },
        {
          "trainer": "Pokemon Trainer May [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Indeedee",
              "item": "Light Clay",
              "moves": [
                "Hyper Voice",
                "Expanding Force",
                "Light Screen",
                "Reflect"
              ],
              "ability": "Psychic Surge"
            },
            {
              "pokemon": "Blaziken",
              "item": "White Herb",
              "moves": [
                "Overheat",
                "Close Combat",
                "Blaze Kick",
                "Stone Edge"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Kingdra",
              "item": "Leftovers",
              "moves": [
                "Liquidation",
                "Scale Shot",
                "Iron Head",
                "Dragon Dance"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Tsareena",
              "item": "Assault Vest",
              "moves": [
                "Trop Kick",
                "Knock Off",
                "Triple Axel",
                "Low Kick"
              ],
              "ability": "Sweet Veil"
            },
            {
              "pokemon": "Weavile",
              "item": "Focus Sash",
              "moves": [
                "Night Slash",
                "Pursuit",
                "Icicle Spear",
                "Low Kick"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Alcremie",
              "item": "Psychic Seed",
              "moves": [
                "Dazzling Gleam",
                "Stored Power",
                "Acid Armor",
                "Recover"
              ],
              "ability": "Sweet Veil"
            }
          ]
        },
        {
          "trainer": "Pokemon Trainer May [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Indeedee",
              "item": "Light Clay",
              "moves": [
                "Hyper Voice",
                "Expanding Force",
                "Light Screen",
                "Reflect"
              ],
              "ability": "Psychic Surge"
            },
            {
              "pokemon": "Swampert",
              "item": "Rindo Berry",
              "moves": [
                "Earthquake",
                "Liquidation",
                "Avalanche",
                "Curse"
              ],
              "ability": "Torrent"
            },
            {
              "pokemon": "Tsareena",
              "item": "Assault Vest",
              "moves": [
                "Trop Kick",
                "Knock Off",
                "Triple Axel",
                "Low Kick"
              ],
              "ability": "Sweet Veil"
            },
            {
              "pokemon": "Houndoom",
              "item": "Power Herb",
              "moves": [
                "Flamethrower",
                "Dark Pulse",
                "Pursuit",
                "Solar Beam"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Hawlucha",
              "item": "Psychic Seed",
              "moves": [
                "Close Combat",
                "Acrobatics",
                "Zen Headbutt",
                "Swords Dance"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Gardevoir",
              "item": "Focus Sash",
              "moves": [
                "Moonblast",
                "Psyshock",
                "Icy Wind",
                "Destiny Bond"
              ],
              "ability": "Synchronize"
            }
          ]
        },
        {
          "trainer": "Pok\u00e9fan Isabel [Double Battle With Pok\u00e9fan Kaleb]",
          "pokemon_list": [
            {
              "pokemon": "Lycanroc_Dusk",
              "item": "Focus Sash",
              "moves": [
                "Rock Slide",
                "Accelerock",
                "Stomping Tantrum",
                "Howl"
              ],
              "ability": "Tough Claws"
            },
            {
              "pokemon": "Victreebel",
              "item": "Salac Berry",
              "moves": [
                "Belch",
                "Leaf Blade",
                "Poison Jab",
                "Sleep Powder"
              ],
              "ability": "Gluttony"
            },
            {
              "pokemon": "Kangaskhan",
              "item": "Sitrus Berry",
              "moves": [
                "Return",
                "Fake Out",
                "Hammer Arm",
                "Sucker Punch"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Fearow",
              "item": "Lum Berry",
              "moves": [
                "Return",
                "Drill Peck",
                "Drill Run",
                "Assurance"
              ],
              "ability": "Sniper"
            }
          ]
        },
        {
          "trainer": "Guitarist Brian",
          "pokemon_list": [
            {
              "pokemon": "Exploud",
              "item": "Choice Scarf",
              "moves": [
                "Boomburst"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Chatot",
              "item": "Lum Berry",
              "moves": [
                "Boomburst",
                "Chatter",
                "Heat Wave",
                "Sing"
              ],
              "ability": "Keen Eye"
            },
            {
              "pokemon": "Heliolisk",
              "item": "Lum Berry",
              "moves": [
                "Thunderbolt",
                "Hyper Voice",
                "Signal Beam",
                "Grass Knot"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Snorlax",
              "item": "Leftovers",
              "moves": [
                "Body Slam",
                "Earthquake",
                "Rest",
                "Sleep Talk"
              ],
              "ability": "Immunity"
            }
          ]
        },
        {
          "trainer": "Collector Edwin",
          "pokemon_list": [
            {
              "pokemon": "Typhlosion",
              "item": "Charcoal",
              "moves": [
                "Eruption",
                "Flamethrower",
                "Focus Blast",
                "Wild Charge"
              ],
              "ability": "Blaze"
            },
            {
              "pokemon": "Feraligatr",
              "item": "Mystic Water",
              "moves": [
                "Liquidation",
                "Crunch",
                "Scary Face",
                "Swords Dance"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Meganium",
              "item": "Miracle Seed",
              "moves": [
                "Seed Bomb",
                "Earthquake",
                "Body Slam",
                "Swords Dance"
              ],
              "ability": "Thick Fat"
            }
          ]
        }
      ]
    },
    "Route 103 (East)": {
      "zone_name": "Route 103 (East)",
      "zone_trainers": [
        {
          "trainer": "Black Belt Rhett [Double Battle With Guitarist Marcos]",
          "pokemon_list": [
            {
              "pokemon": "Crabominable",
              "item": "Lum Berry",
              "moves": [
                "Close Combat",
                "Ice Punch",
                "Crabhammer",
                "Protect"
              ],
              "ability": "Hyper Cutter"
            },
            {
              "pokemon": "Heracross",
              "item": "Flame Orb",
              "moves": [
                "Leech Life",
                "Brick Break",
                "Facade",
                "Protect"
              ],
              "ability": "Guts"
            },
            {
              "pokemon": "Poliwrath",
              "item": "Sitrus Berry",
              "moves": [
                "Liquidation",
                "Drain Punch",
                "Ice Punch",
                "Belly Drum"
              ],
              "ability": "Water Absorb"
            },
            {
              "pokemon": "Crobat",
              "item": "Lum Berry",
              "moves": [
                "Dual Wingbeat",
                "Super Fang",
                "Hypnosis",
                "Tailwind"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Noivern",
              "item": "Lum Berry",
              "moves": [
                "Dragon Pulse",
                "Air Slash",
                "Flamethrower",
                "Tailwind"
              ],
              "ability": "Infiltrator"
            }
          ]
        },
        {
          "trainer": "Pok\u00e9fan Miguel",
          "pokemon_list": [
            {
              "pokemon": "Dedenne",
              "item": "Focus Sash",
              "moves": [
                "Thunderbolt",
                "Dazzling Gleam",
                "Super Fang",
                "Electric Terrain"
              ],
              "ability": "Cheek Pouch"
            },
            {
              "pokemon": "Togedemaru",
              "item": "Rocky Helmet",
              "moves": [
                "Steel Roller",
                "Iron Head",
                "Zing Zap",
                "Magnet Rise"
              ],
              "ability": "Iron Barbs"
            },
            {
              "pokemon": "Mimikyu",
              "item": "Lum Berry",
              "moves": [
                "Play Rough",
                "Shadow Sneak",
                "Drain Punch",
                "Swords Dance"
              ],
              "ability": "Disguise"
            }
          ]
        },
        {
          "trainer": "Aroma Lady Daisy",
          "pokemon_list": [
            {
              "pokemon": "Gourgeist_Super",
              "item": "Lum Berry",
              "moves": [
                "Seed Bomb",
                "Shadow Claw",
                "Explosion",
                "Trick Room"
              ],
              "ability": "Insomnia"
            },
            {
              "pokemon": "Lurantis",
              "item": "Assault Vest",
              "moves": [
                "Leaf Storm",
                "Superpower",
                "Leaf Blade",
                "Facade"
              ],
              "ability": "Contrary"
            },
            {
              "pokemon": "Exeggutor_Alolan",
              "item": "Sitrus Berry",
              "moves": [
                "Wood Hammer",
                "Dragon Hammer",
                "Sleep Powder",
                "Trick Room"
              ],
              "ability": "Harvest"
            },
            {
              "pokemon": "Florges_Orange_Flower",
              "item": "Leftovers",
              "moves": [
                "Dazzling Gleam",
                "Psychic",
                "Giga Drain",
                "Synthesis"
              ],
              "ability": "Flower Veil"
            }
          ]
        },
        {
          "trainer": "Twins Amy & Liv [Double]",
          "pokemon_list": [
            {
              "pokemon": "Gengar",
              "item": "Black Sludge",
              "moves": [
                "Dream Eater",
                "Hex",
                "Substitute",
                "Hypnosis"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Corviknight",
              "item": "Light Clay",
              "moves": [
                "Pluck",
                "Light Screen",
                "Reflect",
                "Roost"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Persian",
              "item": "Focus Sash",
              "moves": [
                "Fake Out",
                "Flail",
                "Dream Eater",
                "Hypnosis"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Jynx",
              "item": "Lum Berry",
              "moves": [
                "Dream Eater",
                "Ice Beam",
                "Nightmare",
                "Lovely Kiss"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Hypno",
              "item": "Leftovers",
              "moves": [
                "Dream Eater",
                "Nightmare",
                "Dark Void"
              ],
              "ability": "Bad Dreams"
            }
          ]
        },
        {
          "trainer": "Fisherman Andrew",
          "pokemon_list": [
            {
              "pokemon": "Jellicent",
              "item": "Lum Berry",
              "moves": [
                "Scald",
                "Shadow Ball",
                "Wring Out",
                "Strength Sap"
              ],
              "ability": "Cursed Body"
            },
            {
              "pokemon": "Barraskewda",
              "item": "Ice Gem",
              "moves": [
                "Liquidation",
                "Drill Run",
                "Throat Chop",
                "Ice Fang"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Bruxish",
              "item": "Water Gem",
              "moves": [
                "Psychic Fangs",
                "Aqua Jet",
                "Crunch",
                "Poison Fang"
              ],
              "ability": "Strong Jaw"
            },
            {
              "pokemon": "Dhelmise",
              "item": "Lagging Tail",
              "moves": [
                "Grassy Glide",
                "Shadow Claw",
                "Iron Head",
                "Switcheroo"
              ],
              "ability": "Steelworker"
            }
          ]
        }
      ]
    },
    "Petalburg Gym": {
      "zone_name": "Petalburg Gym",
      "zone_trainers": [
        {
          "trainer": "Cool Trainer Mary",
          "pokemon_list": [
            {
              "pokemon": "Doublade",
              "item": "Eviolite",
              "moves": [
                "Iron Head",
                "Shadow Claw",
                "Head Smash",
                "Sacred Sword"
              ],
              "ability": "No Guard"
            },
            {
              "pokemon": "Golurk",
              "item": "Lum Berry",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Mega Kick",
                "Dynamic Punch"
              ],
              "ability": "No Guard"
            },
            {
              "pokemon": "Lycanroc_Midnight",
              "item": "Lum Berry",
              "moves": [
                "Stone Edge",
                "Mega Kick",
                "Iron Tail",
                "Fire Fang"
              ],
              "ability": "No Guard"
            },
            {
              "pokemon": "Machamp",
              "item": "Leftovers",
              "moves": [
                "Dynamic Punch",
                "Mega Kick",
                "Stone Edge",
                "Bulk Up"
              ],
              "ability": "No Guard"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Randall",
          "pokemon_list": [
            {
              "pokemon": "Ninjask",
              "item": "Power Herb",
              "moves": [
                "X Scissor",
                "Dual Wingbeat",
                "Dig",
                "Swords Dance"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Scolipede",
              "item": "Muscle Band",
              "moves": [
                "Megahorn",
                "Poison Jab",
                "Aqua Tail",
                "Rock Slide"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Yanmega",
              "item": "Wise Glasses",
              "moves": [
                "Bug Buzz",
                "Air Slash",
                "Giga Drain",
                "Detect"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Sharpedo",
              "item": "Life Orb",
              "moves": [
                "Liquidation",
                "Crunch",
                "Psychic Fangs",
                "Protect"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Blaziken",
              "item": "Focus Sash",
              "moves": [
                "Blaze Kick",
                "Reversal",
                "Thunder Punch",
                "Night Slash"
              ],
              "ability": "Speed Boost"
            }
          ]
        },
        {
          "trainer": "Cool Trainer George",
          "pokemon_list": [
            {
              "pokemon": "Rillaboom",
              "item": "Leftovers",
              "moves": [
                "Grassy Glide",
                "Knock Off",
                "U Turn",
                "Protect"
              ],
              "ability": "Grassy Surge"
            },
            {
              "pokemon": "Arcanine",
              "item": "Leftovers",
              "moves": [
                "Flare Blitz",
                "Play Rough",
                "Wild Charge",
                "Morning Sun"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Starmie",
              "item": "Leftovers",
              "moves": [
                "Psyshock",
                "Scald",
                "Grass Knot",
                "Recover"
              ],
              "ability": "Natural Cure"
            },
            {
              "pokemon": "Slowking_Galarian",
              "item": "Black Sludge",
              "moves": [
                "Psychic",
                "Sludge Bomb",
                "Protect",
                "Slack Off"
              ],
              "ability": "Regenerator"
            },
            {
              "pokemon": "Clefable",
              "item": "Leftovers",
              "moves": [
                "Moonblast",
                "Flamethrower",
                "Calm Mind",
                "Soft Boiled"
              ],
              "ability": "Magic Guard"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Alexia",
          "pokemon_list": [
            {
              "pokemon": "Mudsdale",
              "item": "Sitrus Berry",
              "moves": [
                "High Horsepower",
                "Body Press",
                "Iron Defense",
                "Stealth Rock"
              ],
              "ability": "Stamina"
            },
            {
              "pokemon": "Ferrothorn",
              "item": "Leftovers",
              "moves": [
                "Power Whip",
                "Body Press",
                "Leech Seed",
                "Iron Defense"
              ],
              "ability": "Iron Barbs"
            },
            {
              "pokemon": "Blastoise",
              "item": "Leftovers",
              "moves": [
                "Scald",
                "Body Press",
                "Iron Defense",
                "Protect"
              ],
              "ability": "Torrent"
            },
            {
              "pokemon": "Corviknight",
              "item": "Lum Berry",
              "moves": [
                "Iron Head",
                "Dual Wingbeat",
                "Body Press",
                "Bulk Up"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Claydol",
              "item": "Weakness Policy",
              "moves": [
                "Drill Run",
                "Stored Power",
                "Body Press",
                "Cosmic Power"
              ],
              "ability": "Levitate"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Parker",
          "pokemon_list": [
            {
              "pokemon": "Barbaracle",
              "item": "Scope Lens",
              "moves": [
                "Stone Edge",
                "Razor Shell",
                "Cross Chop",
                "Shadow Claw"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Honchkrow",
              "item": "Scope Lens",
              "moves": [
                "Night Slash",
                "Sucker Punch",
                "Air Cutter",
                "Heat Wave"
              ],
              "ability": "Super Luck"
            },
            {
              "pokemon": "Inteleon",
              "item": "Lum Berry",
              "moves": [
                "Snipe Shot",
                "Ice Beam",
                "Dark Pulse",
                "Laser Focus"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Octillery",
              "item": "Scope Lens",
              "moves": [
                "Octazooka",
                "Gunk Shot",
                "Bullet Seed",
                "Focus Energy"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Drapion",
              "item": "Scope Lens",
              "moves": [
                "Cross Poison",
                "Night Slash",
                "Earthquake",
                "Aqua Tail"
              ],
              "ability": "Sniper"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Berke",
          "pokemon_list": [
            {
              "pokemon": "Floette_Eternal_Flower",
              "item": "Choice Scarf",
              "moves": [
                "Light Of Ruin"
              ],
              "ability": "Flower Veil"
            },
            {
              "pokemon": "Lucario",
              "item": "Sitrus Berry",
              "moves": [
                "Steel Beam",
                "Aura Sphere",
                "Vacuum Wave",
                "Stone Edge"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Staraptor",
              "item": "Lum Berry",
              "moves": [
                "Brave Bird",
                "Double Edge",
                "Quick Attack",
                "Endeavor"
              ],
              "ability": "Reckless"
            },
            {
              "pokemon": "Bouffalant",
              "item": "Custap Berry",
              "moves": [
                "Head Charge",
                "Wild Charge",
                "Lash Out",
                "Reversal"
              ],
              "ability": "Reckless"
            },
            {
              "pokemon": "Emboar",
              "item": "Choice Scarf",
              "moves": [
                "Flare Blitz",
                "Head Smash",
                "Double Edge",
                "Wild Charge"
              ],
              "ability": "Reckless"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Jody",
          "pokemon_list": [
            {
              "pokemon": "Rampardos",
              "item": "Life Orb",
              "moves": [
                "Rock Slide",
                "Crunch",
                "Zen Headbutt",
                "Fire Punch"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Slaking",
              "item": "Choice Band",
              "moves": [
                "Giga Impact"
              ],
              "ability": "Truant"
            },
            {
              "pokemon": "Medicham",
              "item": "Focus Sash",
              "moves": [
                "High Jump Kick",
                "Zen Headbutt",
                "Reversal",
                "Counter"
              ],
              "ability": "Pure Power"
            },
            {
              "pokemon": "Conkeldurr",
              "item": "Lum Berry",
              "moves": [
                "Close Combat",
                "Mach Punch",
                "Thunder Punch",
                "Ice Punch"
              ],
              "ability": "Iron Fist"
            },
            {
              "pokemon": "Marowak",
              "item": "Thick Club",
              "moves": [
                "Bonemerang",
                "Double Edge",
                "Retaliate",
                "Knock Off"
              ],
              "ability": "Rock Head"
            }
          ]
        },
        {
          "trainer": "Leader Norman [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Porygon2",
              "item": "Eviolite",
              "moves": [
                "Tri Attack",
                "Ice Beam",
                "Recover",
                "Thunder Wave"
              ],
              "ability": "Download"
            },
            {
              "pokemon": "Azumarill",
              "item": "Sitrus Berry",
              "moves": [
                "Play Rough",
                "Waterfall",
                "Aqua Jet",
                "Body Slam"
              ],
              "ability": "Huge Power"
            },
            {
              "pokemon": "Diggersby",
              "item": "Focus Sash",
              "moves": [
                "Earthquake",
                "Body Slam",
                "Quick Attack",
                "Foul Play"
              ],
              "ability": "Huge Power"
            },
            {
              "pokemon": "Meloetta",
              "item": "Lum Berry",
              "moves": [
                "Close Combat",
                "Body Slam",
                "Zen Headbutt",
                "Relic Song"
              ],
              "ability": "Serene Grace"
            },
            {
              "pokemon": "Cinccino",
              "item": "Kings Rock",
              "moves": [
                "Tail Slap",
                "Bullet Seed",
                "Rock Blast",
                "Triple Axel"
              ],
              "ability": "Skill Link"
            },
            {
              "pokemon": "Pidgeot",
              "item": "Pidgeotite",
              "moves": [
                "Hurricane",
                "Hyper Voice",
                "Heat Wave",
                "Hidden Power Grass"
              ],
              "ability": "Keen Eye"
            }
          ]
        }
      ]
    },
    "split_name": "Norman"
  },
  "Flannery": {
    "Winstrate House (Route 111)": {
      "zone_name": "Winstrate House (Route 111)",
      "zone_trainers": [
        {
          "trainer": "Winstrate Victor",
          "pokemon_list": [
            {
              "pokemon": "Toucannon",
              "item": "Lum Berry",
              "moves": [
                "Boomburst",
                "Drill Peck",
                "Bullet Seed",
                "Supersonic"
              ],
              "ability": "Skill Link"
            },
            {
              "pokemon": "Emolga",
              "item": "Lum Berry",
              "moves": [
                "Thunderbolt",
                "Dual Wingbeat",
                "Iron Tail",
                "Roost"
              ],
              "ability": "Motor Drive"
            }
          ]
        },
        {
          "trainer": "Winstrate Victoria",
          "pokemon_list": [
            {
              "pokemon": "Abomasnow",
              "item": "Lum Berry",
              "moves": [
                "Wood Hammer",
                "Ice Beam",
                "Ice Shard",
                "Earthquake"
              ],
              "ability": "Soundproof"
            },
            {
              "pokemon": "Centiskorch",
              "item": "Leftovers",
              "moves": [
                "Fire Lash",
                "Skitter Smack",
                "Coil",
                "Protect"
              ],
              "ability": "White Smoke"
            },
            {
              "pokemon": "Seismitoad",
              "item": "Lum Berry",
              "moves": [
                "Earth Power",
                "Muddy Water",
                "Venoshock",
                "Toxic"
              ],
              "ability": "Water Absorb"
            }
          ]
        },
        {
          "trainer": "Winstrate Vivi",
          "pokemon_list": [
            {
              "pokemon": "Ribombee",
              "item": "Focus Sash",
              "moves": [
                "Dazzling Gleam",
                "Signal Beam",
                "Energy Ball",
                "Stun Spore"
              ],
              "ability": "Shield Dust"
            },
            {
              "pokemon": "Flygon",
              "item": "Lum Berry",
              "moves": [
                "Dragon Claw",
                "Scorching Sands",
                "Flamethrower",
                "Roost"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Sudowoodo",
              "item": "Lum Berry",
              "moves": [
                "Head Smash",
                "Double Edge",
                "Wood Hammer",
                "Submission"
              ],
              "ability": "Rock Head"
            }
          ]
        },
        {
          "trainer": "Winstrate Vicky",
          "pokemon_list": [
            {
              "pokemon": "Medicham",
              "item": "Medichamite",
              "moves": [
                "High Jump Kick",
                "Zen Headbutt",
                "Thunder Punch",
                "Fake Out"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Krookodile",
              "item": "Lum Berry",
              "moves": [
                "Earthquake",
                "Crunch",
                "Pursuit",
                "Aqua Tail"
              ],
              "ability": "Moxie"
            }
          ]
        }
      ]
    },
    "Route 111 (Desert), permanent Sandstorm": {
      "zone_name": "Route 111 (Desert), permanent Sandstorm",
      "zone_trainers": [
        {
          "trainer": "Picnicker Irene",
          "pokemon_list": [
            {
              "pokemon": "Leavanny",
              "item": "Focus Sash",
              "moves": [
                "Leaf Blade",
                "X Scissor",
                "Flail",
                "Grass Whistle"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Clefable",
              "item": "Leftovers",
              "moves": [
                "Moonblast",
                "Focus Blast",
                "Counter",
                "Soft Boiled"
              ],
              "ability": "Magic Guard"
            },
            {
              "pokemon": "Stoutland",
              "item": "Silk Scarf",
              "moves": [
                "Retaliate",
                "Body Slam",
                "Play Rough",
                "Stomping Tantrum"
              ],
              "ability": "Sand Rush"
            }
          ]
        },
        {
          "trainer": "Camper Travis",
          "pokemon_list": [
            {
              "pokemon": "Crustle",
              "item": "Lum Berry",
              "moves": [
                "Stone Edge",
                "X Scissor",
                "Earthquake",
                "Shell Smash"
              ],
              "ability": "Weak Armor"
            },
            {
              "pokemon": "Sigilyph",
              "item": "Leftovers",
              "moves": [
                "Psychic",
                "Air Slash",
                "Heat Wave",
                "Energy Ball"
              ],
              "ability": "Magic Guard"
            },
            {
              "pokemon": "Piloswine",
              "item": "Eviolite",
              "moves": [
                "Earthquake",
                "Icicle Spear",
                "Curse",
                "Rest"
              ],
              "ability": "Thick Fat"
            }
          ]
        },
        {
          "trainer": "Ruin Maniac Bryan [Double Battle With Picnicker Celia]",
          "pokemon_list": [
            {
              "pokemon": "Gigalith",
              "item": "Sitrus Berry",
              "moves": [
                "Stone Edge",
                "Rock Slide",
                "Body Press",
                "Sand Tomb"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Nidoking",
              "item": "Lum Berry",
              "moves": [
                "Drill Run",
                "Poison Jab",
                "Thunderbolt",
                "Flamethrower"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Maractus",
              "item": "Sitrus Berry",
              "moves": [
                "Energy Ball",
                "Hidden Power Fire",
                "Weather Ball",
                "Spiky Shield"
              ],
              "ability": "Storm Drain"
            },
            {
              "pokemon": "Gastrodon",
              "item": "Rindo Berry",
              "moves": [
                "Earth Power",
                "Muddy Water",
                "Ice Beam",
                "Recover"
              ],
              "ability": "Storm Drain"
            }
          ]
        },
        {
          "trainer": "Camper Branden",
          "pokemon_list": [
            {
              "pokemon": "Skarmory",
              "item": "Red Card",
              "moves": [
                "Dual Wingbeat",
                "Body Press",
                "Spikes",
                "Stealth Rock"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Torterra",
              "item": "Leftovers",
              "moves": [
                "Wood Hammer",
                "Earthquake",
                "Leech Seed",
                "Protect"
              ],
              "ability": "Rock Head"
            },
            {
              "pokemon": "Bewear",
              "item": "Leftovers",
              "moves": [
                "Body Slam",
                "Drain Punch",
                "Zen Headbutt",
                "Payback"
              ],
              "ability": "Fluffy"
            },
            {
              "pokemon": "Rhyperior",
              "item": "Leftovers",
              "moves": [
                "Drill Run",
                "Avalanche",
                "Dragon Tail",
                "Protect"
              ],
              "ability": "Solid Rock"
            }
          ]
        },
        {
          "trainer": "Collector John",
          "pokemon_list": [
            {
              "pokemon": "Copperajah",
              "item": "Choice Band",
              "moves": [
                "Heavy Slam"
              ],
              "ability": "Heavy Metal"
            },
            {
              "pokemon": "Delphox",
              "item": "Sticky Barb",
              "moves": [
                "Psyshock",
                "Mystical Fire",
                "Foul Play",
                "Switcheroo"
              ],
              "ability": "Magic Guard"
            },
            {
              "pokemon": "Mandibuzz",
              "item": "Rocky Helmet",
              "moves": [
                "Knock Off",
                "Protect",
                "Roost",
                "Toxic"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Lycanroc",
              "item": "Focus Sash",
              "moves": [
                "Stone Edge",
                "High Horsepower",
                "Psychic Fangs",
                "Endeavor"
              ],
              "ability": "Sand Rush"
            },
            {
              "pokemon": "Sandslash",
              "item": "Lum Berry",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Knock Off",
                "Swords Dance"
              ],
              "ability": "Sand Rush"
            }
          ]
        },
        {
          "trainer": "Camper Beau",
          "pokemon_list": [
            {
              "pokemon": "Probopass",
              "item": "Focus Sash",
              "moves": [
                "Flash Cannon",
                "Power Gem",
                "Explosion",
                "Zap Cannon"
              ],
              "ability": "Sand Force"
            },
            {
              "pokemon": "Palossand",
              "item": "Bright Powder",
              "moves": [
                "Scorching Sands",
                "Hex",
                "Ancient Power",
                "Shore Up"
              ],
              "ability": "Sand Veil"
            },
            {
              "pokemon": "Perrserker",
              "item": "Assault Vest",
              "moves": [
                "Iron Tail",
                "Bullet Punch",
                "Close Combat",
                "Gunk Shot"
              ],
              "ability": "Steely Spirit"
            }
          ]
        },
        {
          "trainer": "Ninja Boy Jinra",
          "pokemon_list": [
            {
              "pokemon": "Nidoqueen",
              "item": "Life Orb",
              "moves": [
                "Earth Power",
                "Sludge Bomb",
                "Flamethrower",
                "Ice Beam"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Charizard",
              "item": "Rock Gem",
              "moves": [
                "Flamethrower",
                "Dual Wingbeat",
                "Dragon Claw",
                "Weather Ball"
              ],
              "ability": "Blaze"
            },
            {
              "pokemon": "Golem",
              "item": "Bright Powder",
              "moves": [
                "Earthquake",
                "Rock Slide",
                "Focus Punch",
                "Seismic Toss"
              ],
              "ability": "Sand Veil"
            },
            {
              "pokemon": "Cacturne",
              "item": "Bright Powder",
              "moves": [
                "Seed Bomb",
                "Payback",
                "Focus Punch",
                "Substitute"
              ],
              "ability": "Sand Veil"
            }
          ]
        },
        {
          "trainer": "Ruin Maniac Rigger",
          "pokemon_list": [
            {
              "pokemon": "Dragapult",
              "item": "Light Clay",
              "moves": [
                "Hex",
                "Dragon Breath",
                "Light Screen",
                "Reflect"
              ],
              "ability": "Cursed Body"
            },
            {
              "pokemon": "Krookodile",
              "item": "Weakness Policy",
              "moves": [
                "Earthquake",
                "Power Trip",
                "Rock Slide",
                "Bulk Up"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Cloyster",
              "item": "Lum Berry",
              "moves": [
                "Liquidation",
                "Frost Breath",
                "Weather Ball",
                "Shell Smash"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Shelgon",
              "item": "Eviolite",
              "moves": [
                "Dragon Rush",
                "Brick Break",
                "Facade",
                "Dragon Dance"
              ],
              "ability": "Overcoat"
            }
          ]
        },
        {
          "trainer": "Camper Drew",
          "pokemon_list": [
            {
              "pokemon": "Hippowdon",
              "item": "Lum Berry",
              "moves": [
                "Earthquake",
                "Rock Slide",
                "Curse",
                "Slack Off"
              ],
              "ability": "Sand Force"
            },
            {
              "pokemon": "Gogoat",
              "item": "Leftovers",
              "moves": [
                "Horn Leech",
                "Earthquake",
                "Rock Slide",
                "Leech Seed"
              ],
              "ability": "Sap Sipper"
            },
            {
              "pokemon": "Golem_Alolan",
              "item": "Lum Berry",
              "moves": [
                "Head Smash",
                "Wild Charge",
                "Explosion",
                "Earthquake"
              ],
              "ability": "Rock Head"
            },
            {
              "pokemon": "Excadrill",
              "item": "Air Balloon",
              "moves": [
                "High Horsepower",
                "Iron Head",
                "Rock Slide",
                "Rock Polish"
              ],
              "ability": "Sand Force"
            }
          ]
        }
      ]
    },
    "Route 111 (North)": {
      "zone_name": "Route 111 (North)",
      "zone_trainers": [
        {
          "trainer": "Cool Trainer Wilton",
          "pokemon_list": [
            {
              "pokemon": "Scizor",
              "item": "Occa Berry",
              "moves": [
                "U Turn",
                "Bullet Punch",
                "Dual Wingbeat",
                "Knock Off"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Tangrowth",
              "item": "Assault Vest",
              "moves": [
                "Leaf Storm",
                "Power Whip",
                "Sludge Bomb",
                "Knock Off"
              ],
              "ability": "Regenerator"
            },
            {
              "pokemon": "Golisopod",
              "item": "Assault Vest",
              "moves": [
                "First Impression",
                "Liquidation",
                "Leech Life",
                "Rock Slide"
              ],
              "ability": "Emergency Exit"
            },
            {
              "pokemon": "Noctowl",
              "item": "Blunder Policy",
              "moves": [
                "Hurricane",
                "Hyper Voice",
                "Heat Wave",
                "Hypnosis"
              ],
              "ability": "Tinted Lens"
            },
            {
              "pokemon": "Dugtrio",
              "item": "Focus Sash",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Night Slash",
                "Hone Claws"
              ],
              "ability": "Arena Trap"
            }
          ]
        },
        {
          "trainer": "Black Belt Daisuke",
          "pokemon_list": [
            {
              "pokemon": "Throh",
              "item": "Sitrus Berry",
              "moves": [
                "Circle Throw",
                "Facade",
                "Bulk Up",
                "Recover"
              ],
              "ability": "Guts"
            },
            {
              "pokemon": "Heracross",
              "item": "Flame Orb",
              "moves": [
                "Close Combat",
                "Megahorn",
                "Retaliate",
                "Facade"
              ],
              "ability": "Guts"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Brooke",
          "pokemon_list": [
            {
              "pokemon": "Vanilluxe",
              "item": "Lum Berry",
              "moves": [
                "Blizzard",
                "Freeze Dry",
                "Ice Shard",
                "Explosion"
              ],
              "ability": "Snow Warning"
            },
            {
              "pokemon": "Tentacruel",
              "item": "Safety Goggles",
              "moves": [
                "Surf",
                "Poison Jab",
                "Blizzard",
                "Toxic Spikes"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Dewgong",
              "item": "Leftovers",
              "moves": [
                "Blizzard",
                "Drill Run",
                "Protect",
                "Substitute"
              ],
              "ability": "Ice Body"
            },
            {
              "pokemon": "Magmortar",
              "item": "Safety Goggles",
              "moves": [
                "Flamethrower",
                "Fire Spin",
                "Scorching Sands",
                "Weather Ball"
              ],
              "ability": "Flame Body"
            },
            {
              "pokemon": "Beartic",
              "item": "White Herb",
              "moves": [
                "Icicle Spear",
                "Superpower",
                "Rock Slide",
                "Aqua Jet"
              ],
              "ability": "Slush Rush"
            }
          ]
        }
      ]
    },
    "Route 113": {
      "zone_name": "Route 113",
      "zone_trainers": [
        {
          "trainer": "Pok\u00e9maniac Wyatt",
          "pokemon_list": [
            {
              "pokemon": "Heliolisk",
              "item": "Life Orb",
              "moves": [
                "Hyper Voice",
                "Thunderbolt",
                "Grass Knot",
                "Magnet Rise"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Salazzle",
              "item": "Lum Berry",
              "moves": [
                "Flamethrower",
                "Sludge Bomb",
                "Hidden Power Grass",
                "Nasty Plot"
              ],
              "ability": "Corrosion"
            },
            {
              "pokemon": "Kecleon",
              "item": "Assault Vest",
              "moves": [
                "Drain Punch",
                "Knock Off",
                "Power Up Punch",
                "Shadow Sneak"
              ],
              "ability": "Protean"
            },
            {
              "pokemon": "Dunsparce",
              "item": "Lum Berry",
              "moves": [
                "Headbutt",
                "Coil",
                "Glare",
                "Roost"
              ],
              "ability": "Serene Grace"
            }
          ]
        },
        {
          "trainer": "Ninja Boy Lao",
          "pokemon_list": [
            {
              "pokemon": "Crobat",
              "item": "Life Orb",
              "moves": [
                "Brave Bird",
                "Sludge Bomb",
                "Heat Wave",
                "Giga Drain"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Muk_Alolan",
              "item": "Black Sludge",
              "moves": [
                "Gunk Shot",
                "Knock Off",
                "Ice Punch",
                "Curse"
              ],
              "ability": "Poison Touch"
            },
            {
              "pokemon": "Weezing_Galarian",
              "item": "Black Sludge",
              "moves": [
                "Strange Steam",
                "Venoshock",
                "Flamethrower",
                "Toxic"
              ],
              "ability": "Levitate"
            }
          ]
        },
        {
          "trainer": "Picnicker Sophie",
          "pokemon_list": [
            {
              "pokemon": "Slurpuff",
              "item": "Sitrus Berry",
              "moves": [
                "Dazzling Gleam",
                "Flamethrower",
                "Psychic",
                "Sticky Web"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Lickilicky",
              "item": "Lum Berry",
              "moves": [
                "Double Edge",
                "Power Whip",
                "Brick Break",
                "Knock Off"
              ],
              "ability": "Own Tempo"
            },
            {
              "pokemon": "Appletun",
              "item": "Starf Berry",
              "moves": [
                "Dragon Pulse",
                "Apple Acid",
                "Leech Seed",
                "Recover"
              ],
              "ability": "Gluttony"
            },
            {
              "pokemon": "Alcremie_Caramel_Swirl",
              "item": "Leftovers",
              "moves": [
                "Draining Kiss",
                "Mystical Fire",
                "Acid Armor",
                "Calm Mind"
              ],
              "ability": "Sweet Veil"
            }
          ]
        },
        {
          "trainer": "Ninja Boy Lung",
          "pokemon_list": [
            {
              "pokemon": "Seviper",
              "item": "Shuca Berry",
              "moves": [
                "Sludge Bomb",
                "Flamethrower",
                "Giga Drain",
                "Earthquake"
              ],
              "ability": "Shed Skin"
            },
            {
              "pokemon": "Accelgor",
              "item": "Life Orb",
              "moves": [
                "Bug Buzz",
                "Focus Blast",
                "Water Shuriken",
                "Final Gambit"
              ],
              "ability": "Sticky Hold"
            }
          ]
        },
        {
          "trainer": "Rich Boy Santos",
          "pokemon_list": [
            {
              "pokemon": "Ninetales",
              "item": "Lum Berry",
              "moves": [
                "Flamethrower",
                "Solar Beam",
                "Extrasensory",
                "Hypnosis"
              ],
              "ability": "Drought"
            },
            {
              "pokemon": "Mamoswine",
              "item": "Lum Berry",
              "moves": [
                "Earthquake",
                "Icicle Crash",
                "Ice Shard",
                "Stealth Rock"
              ],
              "ability": "Oblivious"
            },
            {
              "pokemon": "Shiftry",
              "item": "Lum Berry",
              "moves": [
                "Solar Blade",
                "Dark Pulse",
                "Explosion",
                "Heat Wave"
              ],
              "ability": "Chlorophyll"
            },
            {
              "pokemon": "Typhlosion",
              "item": "Choice Scarf",
              "moves": [
                "Eruption"
              ],
              "ability": "Blaze"
            }
          ]
        },
        {
          "trainer": "Twins Tori & Tia [Double]",
          "pokemon_list": [
            {
              "pokemon": "Scrafty",
              "item": "Sitrus Berry",
              "moves": [
                "Close Combat",
                "Knock Off",
                "Foul Play",
                "Fake Out"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Bruxish",
              "item": "Lum Berry",
              "moves": [
                "Psychic Fangs",
                "Liquidation",
                "Crunch",
                "Bulk Up"
              ],
              "ability": "Dazzling"
            },
            {
              "pokemon": "Gothitelle",
              "item": "Sitrus Berry",
              "moves": [
                "Psychic",
                "Signal Beam",
                "Fake Out",
                "Hypnosis"
              ],
              "ability": "Shadow Tag"
            },
            {
              "pokemon": "Bisharp",
              "item": "Lum Berry",
              "moves": [
                "Iron Head",
                "Sucker Punch",
                "Knock Off",
                "Low Kick"
              ],
              "ability": "Defiant"
            }
          ]
        },
        {
          "trainer": "Youngster Jaylen",
          "pokemon_list": [
            {
              "pokemon": "Scyther",
              "item": "Choice Band",
              "moves": [
                "U Turn",
                "Dual Wingbeat"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Talonflame",
              "item": "Flying Gem",
              "moves": [
                "Flare Blitz",
                "Acrobatics",
                "Swords Dance",
                "Will O Wisp"
              ],
              "ability": "Flame Body"
            },
            {
              "pokemon": "Tauros",
              "item": "Lum Berry",
              "moves": [
                "Body Slam",
                "Close Combat",
                "Zen Headbutt",
                "Throat Chop"
              ],
              "ability": "Sheer Force"
            }
          ]
        },
        {
          "trainer": "Bird Keeper Coby",
          "pokemon_list": [
            {
              "pokemon": "Golurk",
              "item": "Lum Berry",
              "moves": [
                "Earthquake",
                "Shadow Punch",
                "Hammer Arm",
                "Stealth Rock"
              ],
              "ability": "Iron Fist"
            },
            {
              "pokemon": "Flapple",
              "item": "Starf Berry",
              "moves": [
                "Dragon Pulse",
                "Grav Apple",
                "Acrobatics",
                "Dragon Dance"
              ],
              "ability": "Gluttony"
            },
            {
              "pokemon": "Vikavolt",
              "item": "Lum Berry",
              "moves": [
                "Bug Buzz",
                "Charge Beam",
                "Energy Ball",
                "Agility"
              ],
              "ability": "Levitate"
            }
          ]
        },
        {
          "trainer": "Parasol Lady Madeline",
          "pokemon_list": [
            {
              "pokemon": "Raichu",
              "item": "Lum Berry",
              "moves": [
                "Thunderbolt",
                "Grass Knot",
                "Extreme Speed",
                "Fake Out"
              ],
              "ability": "Lightning Rod"
            },
            {
              "pokemon": "Marowak_Alolan",
              "item": "Thick Club",
              "moves": [
                "Shadow Bone",
                "Fire Punch",
                "Bonemerang",
                "Low Kick"
              ],
              "ability": "Lightning Rod"
            },
            {
              "pokemon": "Rhyperior",
              "item": "Leftovers",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "High Horsepower",
                "Helping Hand"
              ],
              "ability": "Lightning Rod"
            }
          ]
        },
        {
          "trainer": "Camper Lawrence",
          "pokemon_list": [
            {
              "pokemon": "Staraptor",
              "item": "White Herb",
              "moves": [
                "Dual Wingbeat",
                "Quick Attack",
                "Close Combat",
                "Roost"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Lumineon",
              "item": "Focus Sash",
              "moves": [
                "Hydro Pump",
                "Ice Beam",
                "Air Slash",
                "Flail"
              ],
              "ability": "Storm Drain"
            },
            {
              "pokemon": "Gyarados",
              "item": "Leftovers",
              "moves": [
                "Aqua Tail",
                "Bounce",
                "Dragon Dance",
                "Substitute"
              ],
              "ability": "Intimidate"
            }
          ]
        }
      ]
    },
    "Fallarbor": {
      "zone_name": "Fallarbor",
      "zone_trainers": [
        {
          "trainer": "Winstrate Vito [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Alakazam",
              "item": "Focus Sash",
              "moves": [
                "Psychic",
                "Aura Sphere",
                "Shadow Ball",
                "Charge Beam"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Crobat",
              "item": "Lum Berry",
              "moves": [
                "Sludge Bomb",
                "Heat Wave",
                "Giga Drain",
                "Nasty Plot"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Breloom",
              "item": "Fighting Gem",
              "moves": [
                "Sky Uppercut",
                "Mach Punch",
                "Bullet Seed",
                "Spore"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Crawdaunt",
              "item": "Water Gem",
              "moves": [
                "Crabhammer",
                "Knock Off",
                "Aqua Jet",
                "Close Combat"
              ],
              "ability": "Adaptability"
            },
            {
              "pokemon": "Swellow",
              "item": "Life Orb",
              "moves": [
                "Boomburst",
                "Brave Bird",
                "Heat Wave",
                "Quick Attack"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Aggron",
              "item": "Aggronite",
              "moves": [
                "Iron Head",
                "Earthquake",
                "Body Press",
                "Autotomize"
              ],
              "ability": "Sturdy"
            }
          ]
        }
      ]
    },
    "Route 114": {
      "zone_name": "Route 114",
      "zone_trainers": [
        {
          "trainer": "Picnicker Charlotte",
          "pokemon_list": [
            {
              "pokemon": "Dodrio",
              "item": "Choice Band",
              "moves": [
                "Brave Bird",
                "Double Edge"
              ],
              "ability": "Tangled Feet"
            },
            {
              "pokemon": "Ditto",
              "item": "Focus Sash",
              "moves": [
                "Transform"
              ],
              "ability": "Imposter"
            },
            {
              "pokemon": "Leafeon",
              "item": "Leftovers",
              "moves": [
                "Leaf Blade",
                "Knock Off",
                "Grass Whistle",
                "Swords Dance"
              ],
              "ability": "Leaf Guard"
            }
          ]
        },
        {
          "trainer": "Rich Boy Braw",
          "pokemon_list": [
            {
              "pokemon": "Nidoking",
              "item": "Wide Lens",
              "moves": [
                "Earth Power",
                "Blizzard",
                "Thunder",
                "Lovely Kiss"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Honchkrow",
              "item": "Lum Berry",
              "moves": [
                "Brave Bird",
                "Pursuit",
                "Superpower",
                "Heat Wave"
              ],
              "ability": "Super Luck"
            },
            {
              "pokemon": "Mimikyu",
              "item": "Life Orb",
              "moves": [
                "Play Rough",
                "Shadow Claw",
                "Shadow Sneak",
                "Will O Wisp"
              ],
              "ability": "Disguise"
            },
            {
              "pokemon": "Snorlax",
              "item": "Leftovers",
              "moves": [
                "Body Slam",
                "Earthquake",
                "Heat Crash",
                "Curse"
              ],
              "ability": "Immunity"
            }
          ]
        },
        {
          "trainer": "Fisherman Nolan",
          "pokemon_list": [
            {
              "pokemon": "Milotic",
              "item": "Lum Berry",
              "moves": [
                "Scald",
                "Ice Beam",
                "Mirror Coat",
                "Recover"
              ],
              "ability": "Marvel Scale"
            },
            {
              "pokemon": "Starmie",
              "item": "Wise Glasses",
              "moves": [
                "Hydro Pump",
                "Psychic",
                "Thunderbolt",
                "Grass Knot"
              ],
              "ability": "Analytic"
            }
          ]
        },
        {
          "trainer": "Fisherman Kai",
          "pokemon_list": [
            {
              "pokemon": "Politoed",
              "item": "Leftovers",
              "moves": [
                "Scald",
                "Psychic",
                "Protect",
                "Toxic"
              ],
              "ability": "Drizzle"
            },
            {
              "pokemon": "Barraskewda",
              "item": "Mystic Water",
              "moves": [
                "Liquidation",
                "Flip Turn",
                "Aqua Jet",
                "Close Combat"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Armaldo",
              "item": "Lum Berry",
              "moves": [
                "Stone Edge",
                "X Scissor",
                "Liquidation",
                "Swords Dance"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Golisopod",
              "item": "Bug Gem",
              "moves": [
                "First Impression",
                "Liquidation",
                "Leech Life",
                "Aqua Jet"
              ],
              "ability": "Emergency Exit"
            }
          ]
        },
        {
          "trainer": "Fisherman Claude",
          "pokemon_list": [
            {
              "pokemon": "Qwilfish",
              "item": "Focus Sash",
              "moves": [
                "Poison Jab",
                "Waterfall",
                "Explosion",
                "Toxic Spikes"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Vaporeon",
              "item": "Leftovers",
              "moves": [
                "Scald",
                "Ice Beam",
                "Baton Pass",
                "Substitute"
              ],
              "ability": "Water Absorb"
            },
            {
              "pokemon": "Clawitzer",
              "item": "Iapapa Berry",
              "moves": [
                "Water Pulse",
                "Aura Sphere",
                "Dark Pulse",
                "Sleep Talk"
              ],
              "ability": "Mega Launcher"
            },
            {
              "pokemon": "Empoleon",
              "item": "Leftovers",
              "moves": [
                "Waterfall",
                "Flash Cannon",
                "Earthquake",
                "Protect"
              ],
              "ability": "Defiant"
            }
          ]
        },
        {
          "trainer": "Kindler Bernie",
          "pokemon_list": [
            {
              "pokemon": "Rotom_Heat",
              "item": "White Herb",
              "moves": [
                "Overheat",
                "Thunderbolt",
                "Hidden Power Ice",
                "Nasty Plot"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Simisear",
              "item": "Starf Berry",
              "moves": [
                "Fire Blast",
                "Focus Blast",
                "Acrobatics",
                "Substitute"
              ],
              "ability": "Gluttony"
            },
            {
              "pokemon": "Skuntank",
              "item": "Fire Gem",
              "moves": [
                "Sludge Bomb",
                "Knock Off",
                "Explosion",
                "Fire Blast"
              ],
              "ability": "Aftermath"
            },
            {
              "pokemon": "Druddigon",
              "item": "Rocky Helmet",
              "moves": [
                "Dragon Rush",
                "Flamethrower",
                "Iron Head",
                "Glare"
              ],
              "ability": "Rough Skin"
            }
          ]
        },
        {
          "trainer": "Picnicker Angelina",
          "pokemon_list": [
            {
              "pokemon": "Donphan",
              "item": "Custap Berry",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Knock Off",
                "Endeavor"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Gallade",
              "item": "Leftovers",
              "moves": [
                "Zen Headbutt",
                "Drain Punch",
                "Hypnosis",
                "Will O Wisp"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Purugly",
              "item": "Silk Scarf",
              "moves": [
                "Double Edge",
                "Stomping Tantrum",
                "Super Fang",
                "Hypnosis"
              ],
              "ability": "Defiant"
            }
          ]
        },
        {
          "trainer": "Expert Nancy",
          "pokemon_list": [
            {
              "pokemon": "Bronzong",
              "item": "Iron Ball",
              "moves": [
                "Gyro Ball",
                "Explosion",
                "Earthquake",
                "Trick"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Eldegoss",
              "item": "Eject Button",
              "moves": [
                "Leaf Storm",
                "Pollen Puff",
                "Leech Seed",
                "Sleep Powder"
              ],
              "ability": "Cotton Down"
            },
            {
              "pokemon": "Sirfetchd",
              "item": "Leek",
              "moves": [
                "Meteor Assault",
                "Brick Break",
                "Leaf Blade",
                "Night Slash"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Drampa",
              "item": "Lum Berry",
              "moves": [
                "Hyper Voice",
                "Dragon Pulse",
                "Glare",
                "Roost"
              ],
              "ability": "Berserk"
            },
            {
              "pokemon": "Dugtrio",
              "item": "Soft Sand",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Night Slash",
                "Final Gambit"
              ],
              "ability": "Arena Trap"
            }
          ]
        },
        {
          "trainer": "Sr. And Jr. Tyra & Ivy [Double]",
          "pokemon_list": [
            {
              "pokemon": "Klefki",
              "item": "Occa Berry",
              "moves": [
                "Foul Play",
                "Swagger"
              ],
              "ability": "Prankster"
            },
            {
              "pokemon": "Houndoom",
              "item": "Life Orb",
              "moves": [
                "Foul Play",
                "Burning Jealousy",
                "Sludge Bomb",
                "Protect"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Grimmsnarl",
              "item": "Sitrus Berry",
              "moves": [
                "Foul Play",
                "Spirit Break",
                "Fake Out",
                "Swagger"
              ],
              "ability": "Prankster"
            },
            {
              "pokemon": "Pyroar",
              "item": "Normal Gem",
              "moves": [
                "Retaliate",
                "Hyper Voice",
                "Burning Jealousy",
                "Protect"
              ],
              "ability": "Unnerve"
            }
          ]
        },
        {
          "trainer": "Pok\u00e9maniac Steve",
          "pokemon_list": [
            {
              "pokemon": "Muk",
              "item": "Shuca Berry",
              "moves": [
                "Gunk Shot",
                "Explosion",
                "Fire Punch",
                "Ice Punch"
              ],
              "ability": "Poison Touch"
            },
            {
              "pokemon": "Polteageist",
              "item": "White Herb",
              "moves": [
                "Shadow Ball",
                "Giga Drain",
                "Stored Power",
                "Shell Smash"
              ],
              "ability": "Cursed Body"
            }
          ]
        },
        {
          "trainer": "Hiker Lucas",
          "pokemon_list": [
            {
              "pokemon": "Steelix",
              "item": "Iapapa Berry",
              "moves": [
                "Earthquake",
                "Gyro Ball",
                "Body Press",
                "Stealth Rock"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Relicanth",
              "item": "Lum Berry",
              "moves": [
                "Stone Edge",
                "Liquidation",
                "Flail",
                "Rock Polish"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Gliscor",
              "item": "Toxic Orb",
              "moves": [
                "Earthquake",
                "Facade",
                "Protect",
                "Roost"
              ],
              "ability": "Poison Heal"
            }
          ]
        },
        {
          "trainer": "Hiker Lenny",
          "pokemon_list": [
            {
              "pokemon": "Ferrothorn",
              "item": "Occa Berry",
              "moves": [
                "Power Whip",
                "Gyro Ball",
                "Knock Off",
                "Thunder Wave"
              ],
              "ability": "Iron Barbs"
            },
            {
              "pokemon": "Cradily",
              "item": "Leftovers",
              "moves": [
                "Ancient Power",
                "Earth Power",
                "Cosmic Power",
                "Rest"
              ],
              "ability": "Suction Cups"
            },
            {
              "pokemon": "Whiscash",
              "item": "Power Herb",
              "moves": [
                "Earthquake",
                "Aqua Tail",
                "Bounce",
                "Dragon Dance"
              ],
              "ability": "Oblivious"
            }
          ]
        }
      ]
    },
    "Route 115": {
      "zone_name": "Route 115",
      "zone_trainers": [
        {
          "trainer": "Black Belt Nob",
          "pokemon_list": [
            {
              "pokemon": "Mienshao",
              "item": "Life Orb",
              "moves": [
                "High Jump Kick",
                "Stone Edge",
                "Poison Jab",
                "Grass Knot"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Bewear",
              "item": "Sitrus Berry",
              "moves": [
                "Hammer Arm",
                "Body Slam",
                "Retaliate",
                "Counter"
              ],
              "ability": "Fluffy"
            }
          ]
        },
        {
          "trainer": "Battle Girl Cyndy",
          "pokemon_list": [
            {
              "pokemon": "Grapploct",
              "item": "Leftovers",
              "moves": [
                "Drain Punch",
                "Skitter Smack",
                "Octolock",
                "Protect"
              ],
              "ability": "Limber"
            },
            {
              "pokemon": "Breloom",
              "item": "Toxic Orb",
              "moves": [
                "Drain Punch",
                "Facade",
                "Bulk Up",
                "Protect"
              ],
              "ability": "Poison Heal"
            },
            {
              "pokemon": "Hitmontop",
              "item": "Life Orb",
              "moves": [
                "Mach Punch",
                "Bullet Punch",
                "Fake Out",
                "Triple Axel"
              ],
              "ability": "Technician"
            }
          ]
        },
        {
          "trainer": "Psychic Marlene",
          "pokemon_list": [
            {
              "pokemon": "Mr_Rime",
              "item": "Lum Berry",
              "moves": [
                "Ice Beam",
                "Psyshock",
                "Teeter Dance",
                "Thunder Wave"
              ],
              "ability": "Screen Cleaner"
            },
            {
              "pokemon": "Cofagrigus",
              "item": "Leftovers",
              "moves": [
                "Hex",
                "Body Press",
                "Will O Wisp",
                "Rest"
              ],
              "ability": "Mummy"
            }
          ]
        },
        {
          "trainer": "Collector Hector",
          "pokemon_list": [
            {
              "pokemon": "Heatmor",
              "item": "Expert Belt",
              "moves": [
                "Flare Blitz",
                "Focus Blast",
                "Thunder Punch",
                "Grass Knot"
              ],
              "ability": "White Smoke"
            },
            {
              "pokemon": "Durant",
              "item": "Lum Berry",
              "moves": [
                "Iron Head",
                "X Scissor",
                "Rock Slide",
                "Hone Claws"
              ],
              "ability": "Hustle"
            },
            {
              "pokemon": "Slowking",
              "item": "Leftovers",
              "moves": [
                "Psychic",
                "Surf",
                "Slack Off",
                "Toxic"
              ],
              "ability": "Regenerator"
            },
            {
              "pokemon": "Zangoose",
              "item": "Toxic Orb",
              "moves": [
                "Facade",
                "Close Combat",
                "Night Slash",
                "Detect"
              ],
              "ability": "Toxic Boost"
            }
          ]
        }
      ]
    },
    "Route 112 (North)": {
      "zone_name": "Route 112 (North)",
      "zone_trainers": [
        {
          "trainer": "Kindler Bryant [Double Battle With Aroma Lady Shayla]",
          "pokemon_list": [
            {
              "pokemon": "Torkoal",
              "item": "Lum Berry",
              "moves": [
                "Heat Wave",
                "Body Press",
                "Scorching Sands",
                "Stealth Rock"
              ],
              "ability": "Drought"
            },
            {
              "pokemon": "Charizard",
              "item": "Iapapa Berry",
              "moves": [
                "Heat Wave",
                "Flamethrower",
                "Air Slash",
                "Solar Beam"
              ],
              "ability": "Solar Power"
            },
            {
              "pokemon": "Leafeon",
              "item": "Grass Gem",
              "moves": [
                "Solar Blade",
                "Knock Off",
                "Fake Tears",
                "Grass Whistle"
              ],
              "ability": "Chlorophyll"
            },
            {
              "pokemon": "Venusaur",
              "item": "Life Orb",
              "moves": [
                "Petal Blizzard",
                "Sludge Bomb",
                "Earthquake",
                "Weather Ball"
              ],
              "ability": "Chlorophyll"
            }
          ]
        },
        {
          "trainer": "Camper Merc",
          "pokemon_list": [
            {
              "pokemon": "Primeape",
              "item": "Expert Belt",
              "moves": [
                "Close Combat",
                "Stone Edge",
                "Earthquake",
                "Poison Jab"
              ],
              "ability": "Defiant"
            },
            {
              "pokemon": "Drapion",
              "item": "Dark Gem",
              "moves": [
                "Cross Poison",
                "Pursuit",
                "Earthquake",
                "Rock Slide"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Trevenant",
              "item": "Sitrus Berry",
              "moves": [
                "Horn Leech",
                "Shadow Claw",
                "Substitute",
                "Will O Wisp"
              ],
              "ability": "Harvest"
            },
            {
              "pokemon": "Scyther",
              "item": "Eviolite",
              "moves": [
                "Dual Wingbeat",
                "Bug Bite",
                "Brick Break",
                "Swords Dance"
              ],
              "ability": "Technician"
            }
          ]
        }
      ]
    },
    "Route 112 (South)": {
      "zone_name": "Route 112 (South)",
      "zone_trainers": [
        {
          "trainer": "Hiker Trent",
          "pokemon_list": [
            {
              "pokemon": "Bastiodon",
              "item": "Red Card",
              "moves": [
                "Heavy Slam",
                "Metal Burst",
                "Stealth Rock",
                "Roar"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Diggersby",
              "item": "Chople Berry",
              "moves": [
                "High Horsepower",
                "Body Slam",
                "Quick Attack",
                "Swords Dance"
              ],
              "ability": "Huge Power"
            },
            {
              "pokemon": "Magcargo",
              "item": "Red Card",
              "moves": [
                "Flamethrower",
                "Power Gem",
                "Earth Power",
                "Shell Smash"
              ],
              "ability": "Weak Armor"
            }
          ]
        },
        {
          "trainer": "Hiker Brice",
          "pokemon_list": [
            {
              "pokemon": "Bronzong",
              "item": "Colbur Berry",
              "moves": [
                "Future Sight",
                "Gyro Ball",
                "Explosion",
                "Trick Room"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Solrock",
              "item": "Room Service",
              "moves": [
                "Stone Edge",
                "Earthquake",
                "Acrobatics",
                "Trick Room"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Lunatone",
              "item": "Room Service",
              "moves": [
                "Psychic",
                "Power Gem",
                "Explosion",
                "Trick Room"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Camerupt",
              "item": "Fire Gem",
              "moves": [
                "Eruption",
                "Earth Power",
                "Flamethrower",
                "Rock Slide"
              ],
              "ability": "Solid Rock"
            }
          ]
        },
        {
          "trainer": "Picnicker Carol",
          "pokemon_list": [
            {
              "pokemon": "Dubwool",
              "item": "Assault Vest",
              "moves": [
                "Mega Kick",
                "Zen Headbutt",
                "Payback",
                "Counter"
              ],
              "ability": "Fluffy"
            },
            {
              "pokemon": "Ampharos",
              "item": "Shuca Berry",
              "moves": [
                "Thunderbolt",
                "Focus Blast",
                "Hidden Power Ice",
                "Agility"
              ],
              "ability": "Static"
            },
            {
              "pokemon": "Spinda",
              "item": "Chople Berry",
              "moves": [
                "Dizzy Punch",
                "Superpower",
                "Psycho Cut",
                "Hypnosis"
              ],
              "ability": "Contrary"
            }
          ]
        }
      ]
    },
    "Mt. Chimney": {
      "zone_name": "Mt. Chimney",
      "zone_trainers": [
        {
          "trainer": "Team Magma Grunt [Double Battle With Team Magma Grunt]",
          "pokemon_list": [
            {
              "pokemon": "Accelgor",
              "item": "Focus Sash",
              "moves": [
                "Bug Buzz",
                "Struggle Bug",
                "Acid Spray",
                "Encore"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Sawk",
              "item": "Weakness Policy",
              "moves": [
                "Close Combat",
                "Reversal",
                "Zen Headbutt",
                "Throat Chop"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Indeedee_Female",
              "item": "Wise Glasses",
              "moves": [
                "Hyper Voice",
                "Expanding Force"
              ],
              "ability": "Psychic Surge"
            },
            {
              "pokemon": "Meowstic_Female",
              "item": "Fairy Gem",
              "moves": [
                "Expanding Force",
                "Thunderbolt",
                "Dazzling Gleam",
                "Shadow Ball"
              ],
              "ability": "Competitive"
            }
          ]
        },
        {
          "trainer": "Magma Admin Tabitha",
          "pokemon_list": [
            {
              "pokemon": "Torkoal",
              "item": "Quick Claw",
              "moves": [
                "Flamethrower",
                "Explosion",
                "Solar Beam",
                "Earth Power"
              ],
              "ability": "Drought"
            },
            {
              "pokemon": "Armaldo",
              "item": "Leftovers",
              "moves": [
                "Stone Edge",
                "Leech Life",
                "Earthquake",
                "Stealth Rock"
              ],
              "ability": "Battle Armor"
            },
            {
              "pokemon": "Espeon",
              "item": "Life Orb",
              "moves": [
                "Psychic",
                "Dazzling Gleam",
                "Weather Ball",
                "Morning Sun"
              ],
              "ability": "Synchronize"
            },
            {
              "pokemon": "Victreebel",
              "item": "Lum Berry",
              "moves": [
                "Solar Blade",
                "Gunk Shot",
                "Weather Ball",
                "Sleep Powder"
              ],
              "ability": "Chlorophyll"
            },
            {
              "pokemon": "Absol",
              "item": "Absolite",
              "moves": [
                "Knock Off",
                "Psycho Cut",
                "Fire Blast",
                "Ice Beam"
              ],
              "ability": "Pressure"
            }
          ]
        },
        {
          "trainer": "Magma Leader Maxie [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Crustle",
              "item": "Red Card",
              "moves": [
                "Stone Edge",
                "Earthquake",
                "Spikes",
                "Stealth Rock"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Wobbuffet",
              "item": "Iapapa Berry",
              "moves": [
                "Counter",
                "Mirror Coat",
                "Destiny Bond",
                "Encore"
              ],
              "ability": "Shadow Tag"
            },
            {
              "pokemon": "Aerodactyl",
              "item": "Dark Gem",
              "moves": [
                "Stone Edge",
                "Dual Wingbeat",
                "Aqua Tail",
                "Pursuit"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Kommo_O",
              "item": "Leftovers",
              "moves": [
                "Body Press",
                "Dragon Tail",
                "Poison Jab",
                "Iron Defense"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Zarude",
              "item": "Assault Vest",
              "moves": [
                "Power Whip",
                "Darkest Lariat",
                "U Turn",
                "Low Kick"
              ],
              "ability": "Leaf Guard"
            },
            {
              "pokemon": "Camerupt",
              "item": "Cameruptite",
              "moves": [
                "Earth Power",
                "Flamethrower",
                "Rock Slide",
                "Substitute"
              ],
              "ability": "Own Tempo"
            }
          ]
        }
      ]
    },
    "Jagged Pass": {
      "zone_name": "Jagged Pass",
      "zone_trainers": [
        {
          "trainer": "Hiker Eric [Double Battle With Picnicker Autumn]",
          "pokemon_list": [
            {
              "pokemon": "Gliscor",
              "item": "Flying Gem",
              "moves": [
                "High Horsepower",
                "Acrobatics",
                "Knock Off",
                "Roost"
              ],
              "ability": "Hyper Cutter"
            },
            {
              "pokemon": "Stonjourner",
              "item": "Focus Sash",
              "moves": [
                "Stone Edge",
                "Rock Slide",
                "Stomping Tantrum",
                "Protect"
              ],
              "ability": "Power Spot"
            },
            {
              "pokemon": "Ferrothorn",
              "item": "Rocky Helmet",
              "moves": [
                "Power Whip",
                "Gyro Ball",
                "Thunder Wave",
                "Stealth Rock"
              ],
              "ability": "Iron Barbs"
            },
            {
              "pokemon": "Serperior",
              "item": "Choice Scarf",
              "moves": [
                "Leaf Storm"
              ],
              "ability": "Contrary"
            },
            {
              "pokemon": "Boltund",
              "item": "Life Orb",
              "moves": [
                "Thunder Fang",
                "Psychic Fangs",
                "Crunch",
                "Snarl"
              ],
              "ability": "Strong Jaw"
            }
          ]
        },
        {
          "trainer": "Triathlete Julio",
          "pokemon_list": [
            {
              "pokemon": "Zebstrika",
              "item": "Shuca Berry",
              "moves": [
                "Zing Zap",
                "Overheat",
                "Low Kick",
                "Light Screen"
              ],
              "ability": "Sap Sipper"
            },
            {
              "pokemon": "Bibarel",
              "item": "Quick Claw",
              "moves": [
                "Body Slam",
                "Waterfall",
                "Stomping Tantrum",
                "Curse"
              ],
              "ability": "Simple"
            },
            {
              "pokemon": "Klinklang",
              "item": "Lum Berry",
              "moves": [
                "Gear Grind",
                "Return",
                "Wild Charge",
                "Shift Gear"
              ],
              "ability": "Clear Body"
            }
          ]
        },
        {
          "trainer": "Camper Ethan",
          "pokemon_list": [
            {
              "pokemon": "Bouffalant",
              "item": "Chople Berry",
              "moves": [
                "Head Charge",
                "Megahorn",
                "Zen Headbutt",
                "Revenge"
              ],
              "ability": "Reckless"
            },
            {
              "pokemon": "Togekiss",
              "item": "Scope Lens",
              "moves": [
                "Air Cutter",
                "Dazzling Gleam",
                "Aura Sphere",
                "Substitute"
              ],
              "ability": "Super Luck"
            }
          ]
        },
        {
          "trainer": "Picnicker Diana",
          "pokemon_list": [
            {
              "pokemon": "Rillaboom",
              "item": "Assault Vest",
              "moves": [
                "Wood Hammer",
                "Grassy Glide",
                "High Horsepower",
                "Knock Off"
              ],
              "ability": "Grassy Surge"
            },
            {
              "pokemon": "Wigglytuff",
              "item": "Life Orb",
              "moves": [
                "Self Destruct",
                "Moonblast",
                "Grass Knot",
                "Sing"
              ],
              "ability": "Competitive"
            },
            {
              "pokemon": "Arcanine",
              "item": "Leftovers",
              "moves": [
                "Flare Blitz",
                "Wild Charge",
                "Close Combat",
                "Howl"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Thievul",
              "item": "Grassy Seed",
              "moves": [
                "Dark Pulse",
                "Burning Jealousy",
                "Grass Knot",
                "Nasty Plot"
              ],
              "ability": "Unburden"
            }
          ]
        }
      ]
    },
    "Lavaridge Gym": {
      "zone_name": "Lavaridge Gym",
      "zone_trainers": [
        {
          "trainer": "Kindler Jace",
          "pokemon_list": [
            {
              "pokemon": "Cinderace",
              "item": "Wide Lens",
              "moves": [
                "Pyro Ball",
                "High Jump Kick",
                "Gunk Shot",
                "Zen Headbutt"
              ],
              "ability": "Libero"
            },
            {
              "pokemon": "Talonflame",
              "item": "Flying Gem",
              "moves": [
                "Flare Blitz",
                "Acrobatics",
                "Will O Wisp",
                "Roost"
              ],
              "ability": "Flame Body"
            },
            {
              "pokemon": "Rotom_Heat",
              "item": "Leftovers",
              "moves": [
                "Overheat",
                "Thunderbolt",
                "Will O Wisp",
                "Nasty Plot"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Chandelure",
              "item": "Choice Scarf",
              "moves": [
                "Fire Blast",
                "Shadow Ball",
                "Hex",
                "Psychic"
              ],
              "ability": "Shadow Tag"
            }
          ]
        },
        {
          "trainer": "Kindler Cole",
          "pokemon_list": [
            {
              "pokemon": "Torkoal",
              "item": "Iapapa Berry",
              "moves": [
                "Flamethrower",
                "Solar Beam",
                "Scorching Sands",
                "Stealth Rock"
              ],
              "ability": "Drought"
            },
            {
              "pokemon": "Octillery",
              "item": "Scope Lens",
              "moves": [
                "Fire Blast",
                "Gunk Shot",
                "Psychic",
                "Thunder Wave"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Weezing_Galarian",
              "item": "Lum Berry",
              "moves": [
                "Play Rough",
                "Explosion",
                "Payback",
                "Overheat"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Magmortar",
              "item": "Sitrus Berry",
              "moves": [
                "Flamethrower",
                "Solar Beam",
                "Belch",
                "Earthquake"
              ],
              "ability": "Vital Spirit"
            },
            {
              "pokemon": "Turtonator",
              "item": "Lum Berry",
              "moves": [
                "Overheat",
                "Dragon Claw",
                "Heat Crash",
                "Shell Smash"
              ],
              "ability": "Shell Armor"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Gerald",
          "pokemon_list": [
            {
              "pokemon": "Ninetales",
              "item": "Focus Sash",
              "moves": [
                "Flamethrower",
                "Psychic",
                "Hypnosis",
                "Nasty Plot"
              ],
              "ability": "Drought"
            },
            {
              "pokemon": "Solrock",
              "item": "Fire Gem",
              "moves": [
                "Zen Headbutt",
                "Rock Slide",
                "Overheat",
                "Rock Polish"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Toucannon",
              "item": "Fire Gem",
              "moves": [
                "Brave Bird",
                "Overheat",
                "Bullet Seed",
                "Rock Blast"
              ],
              "ability": "Skill Link"
            }
          ]
        },
        {
          "trainer": "Kindler Keegan",
          "pokemon_list": [
            {
              "pokemon": "Rapidash_Galarian",
              "item": "Life Orb",
              "moves": [
                "Play Rough",
                "Zen Headbutt",
                "Mystical Fire",
                "Will O Wisp"
              ],
              "ability": "Pastel Veil"
            },
            {
              "pokemon": "Blissey",
              "item": "Chople Berry",
              "moves": [
                "Hyper Voice",
                "Fire Blast",
                "Psychic",
                "Counter"
              ],
              "ability": "Serene Grace"
            },
            {
              "pokemon": "Darmanitan",
              "item": "Life Orb",
              "moves": [
                "Flare Blitz",
                "Zen Headbutt",
                "Rock Slide",
                "Earthquake"
              ],
              "ability": "Sheer Force"
            }
          ]
        },
        {
          "trainer": "Kindler Axle",
          "pokemon_list": [
            {
              "pokemon": "Infernape",
              "item": "Focus Sash",
              "moves": [
                "Fire Blast",
                "Focus Blast",
                "Grass Knot",
                "Fake Out"
              ],
              "ability": "Blaze"
            },
            {
              "pokemon": "Houndoom",
              "item": "Lum Berry",
              "moves": [
                "Fire Blast",
                "Dark Pulse",
                "Pursuit",
                "Destiny Bond"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Centiskorch",
              "item": "Charti Berry",
              "moves": [
                "Fire Lash",
                "Leech Life",
                "Thunder Fang",
                "Coil"
              ],
              "ability": "White Smoke"
            }
          ]
        },
        {
          "trainer": "Kindler Jeff",
          "pokemon_list": [
            {
              "pokemon": "Torkoal",
              "item": "Custap Berry",
              "moves": [
                "Fire Blast",
                "Explosion",
                "Earth Power",
                "Body Press"
              ],
              "ability": "Drought"
            },
            {
              "pokemon": "Lickilicky",
              "item": "Assault Vest",
              "moves": [
                "Explosion",
                "Body Slam",
                "Fire Blast",
                "Power Whip"
              ],
              "ability": "Oblivious"
            },
            {
              "pokemon": "Skuntank",
              "item": "Lum Berry",
              "moves": [
                "Poison Jab",
                "Knock Off",
                "Explosion",
                "Fire Blast"
              ],
              "ability": "Aftermath"
            },
            {
              "pokemon": "Flareon",
              "item": "Toxic Orb",
              "moves": [
                "Flare Blitz",
                "Solar Beam",
                "Facade",
                "Stomping Tantrum"
              ],
              "ability": "Guts"
            }
          ]
        },
        {
          "trainer": "Battle Girl Danielle",
          "pokemon_list": [
            {
              "pokemon": "Slowking_Galarian",
              "item": "Assault Vest",
              "moves": [
                "Future Sight",
                "Psychic",
                "Fire Blast",
                "Scald"
              ],
              "ability": "Regenerator"
            },
            {
              "pokemon": "Golurk",
              "item": "Leftovers",
              "moves": [
                "Earthquake",
                "Shadow Punch",
                "Heat Crash",
                "Block"
              ],
              "ability": "Iron Fist"
            },
            {
              "pokemon": "Emboar",
              "item": "Choice Scarf",
              "moves": [
                "Flare Blitz",
                "Superpower",
                "Head Smash",
                "Wild Charge"
              ],
              "ability": "Reckless"
            },
            {
              "pokemon": "Blaziken",
              "item": "White Herb",
              "moves": [
                "Overheat",
                "Close Combat",
                "Brave Bird",
                "Knock Off"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Flygon",
              "item": "Life Orb",
              "moves": [
                "Earthquake",
                "Scale Shot",
                "Fire Blast",
                "Giga Drain"
              ],
              "ability": "Levitate"
            }
          ]
        },
        {
          "trainer": "Hiker Eli",
          "pokemon_list": [
            {
              "pokemon": "Drampa",
              "item": "Iapapa Berry",
              "moves": [
                "Hyper Voice",
                "Dragon Pulse",
                "Fire Blast",
                "Roost"
              ],
              "ability": "Berserk"
            },
            {
              "pokemon": "Nidoking",
              "item": "Black Sludge",
              "moves": [
                "Sludge Bomb",
                "Earth Power",
                "Fire Blast",
                "Substitute"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Rhyperior",
              "item": "Passho Berry",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Heat Crash",
                "Stealth Rock"
              ],
              "ability": "Solid Rock"
            },
            {
              "pokemon": "Arcanine",
              "item": "Fire Gem",
              "moves": [
                "Burn Up",
                "Close Combat",
                "Psychic Fangs",
                "Will O Wisp"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Delphox",
              "item": "Life Orb",
              "moves": [
                "Flamethrower",
                "Psychic",
                "Grass Knot",
                "Hypnosis"
              ],
              "ability": "Magic Guard"
            }
          ]
        },
        {
          "trainer": "Leader Flannery [Double] [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Charizard",
              "item": "Charizardite Y",
              "moves": [
                "Fire Blast",
                "Heat Wave",
                "Air Slash",
                "Solar Beam"
              ],
              "ability": "Blaze"
            },
            {
              "pokemon": "Salazzle",
              "item": "Focus Sash",
              "moves": [
                "Heat Wave",
                "Sludge Bomb",
                "Fake Out",
                "Endeavor"
              ],
              "ability": "Corrosion"
            },
            {
              "pokemon": "Entei",
              "item": "Lum Berry",
              "moves": [
                "Sacred Fire",
                "Solar Beam",
                "Stone Edge",
                "Extreme Speed"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Incineroar",
              "item": "Assault Vest",
              "moves": [
                "Overheat",
                "Blaze Kick",
                "Darkest Lariat",
                "Fake Out"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Talonflame",
              "item": "Life Orb",
              "moves": [
                "Brave Bird",
                "Flare Blitz",
                "Solar Beam",
                "Tailwind"
              ],
              "ability": "Gale Wings"
            },
            {
              "pokemon": "Marowak_Alolan",
              "item": "Thick Club",
              "moves": [
                "Flare Blitz",
                "Poltergeist",
                "Bonemerang",
                "Detect"
              ],
              "ability": "Rock Head"
            }
          ]
        }
      ]
    },
    "split_name": "Flannery"
  },
  "Winona": {
    "Seashore House (Route 109)": {
      "zone_name": "Seashore House (Route 109)",
      "zone_trainers": [
        {
          "trainer": "Tuber Simon",
          "pokemon_list": [
            {
              "pokemon": "Floatzel",
              "item": "Flame Orb",
              "moves": [
                "Aqua Tail",
                "Ice Beam",
                "Knock Off",
                "Switcheroo"
              ],
              "ability": "Water Veil"
            },
            {
              "pokemon": "Azumarill",
              "item": "Assault Vest",
              "moves": [
                "Play Rough",
                "Aqua Tail",
                "Aqua Jet",
                "Ice Punch"
              ],
              "ability": "Huge Power"
            },
            {
              "pokemon": "Dedenne",
              "item": "Petaya Berry",
              "moves": [
                "Thunderbolt",
                "Dazzling Gleam",
                "Grass Knot",
                "Endure"
              ],
              "ability": "Cheek Pouch"
            },
            {
              "pokemon": "Zangoose",
              "item": "Toxic Orb",
              "moves": [
                "Last Resort",
                "Retaliate"
              ],
              "ability": "Toxic Boost"
            },
            {
              "pokemon": "Persian",
              "item": "Leftovers",
              "moves": [
                "Hyper Voice",
                "Hypnosis",
                "Nasty Plot",
                "Substitute"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Ludicolo",
              "item": "Lum Berry",
              "moves": [
                "Waterfall",
                "Seed Bomb",
                "Earthquake",
                "Swords Dance"
              ],
              "ability": "Own Tempo"
            }
          ]
        },
        {
          "trainer": "Beauty Johanna",
          "pokemon_list": [
            {
              "pokemon": "Nidoqueen",
              "item": "Black Sludge",
              "moves": [
                "Sludge Wave",
                "Earth Power",
                "Lovely Kiss",
                "Toxic Spikes"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Lapras",
              "item": "Leftovers",
              "moves": [
                "Freeze Dry",
                "Thunderbolt",
                "Protect",
                "Roar"
              ],
              "ability": "Shell Armor"
            },
            {
              "pokemon": "Milotic",
              "item": "Leftovers",
              "moves": [
                "Aqua Tail",
                "Dragon Tail",
                "Coil",
                "Rest"
              ],
              "ability": "Marvel Scale"
            },
            {
              "pokemon": "Primarina",
              "item": "Leftovers",
              "moves": [
                "Scald",
                "Draining Kiss",
                "Calm Mind",
                "Protect"
              ],
              "ability": "Torrent"
            },
            {
              "pokemon": "Bruxish",
              "item": "Water Gem",
              "moves": [
                "Aqua Tail",
                "Psychic Fangs",
                "Crunch",
                "Screech"
              ],
              "ability": "Strong Jaw"
            },
            {
              "pokemon": "Tsareena",
              "item": "Wide Lens",
              "moves": [
                "Power Whip",
                "High Jump Kick",
                "Knock Off",
                "Triple Axel"
              ],
              "ability": "Queenly Majesty"
            }
          ]
        },
        {
          "trainer": "Sailor Dwayne",
          "pokemon_list": [
            {
              "pokemon": "Clawitzer",
              "item": "Lum Berry",
              "moves": [
                "Water Pulse",
                "Dragon Pulse",
                "Aura Sphere",
                "Aqua Jet"
              ],
              "ability": "Mega Launcher"
            },
            {
              "pokemon": "Gyarados",
              "item": "Wacan Berry",
              "moves": [
                "Waterfall",
                "Earthquake",
                "Lash Out",
                "Thunder Wave"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Barbaracle",
              "item": "White Herb",
              "moves": [
                "Stone Edge",
                "Razor Shell",
                "Cross Chop",
                "Shell Smash"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Sharpedo",
              "item": "Dark Gem",
              "moves": [
                "Hydro Pump",
                "Dark Pulse",
                "Close Combat",
                "Destiny Bond"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Swalot",
              "item": "Custap Berry",
              "moves": [
                "Gunk Shot",
                "Explosion",
                "Earthquake",
                "Curse"
              ],
              "ability": "Gluttony"
            },
            {
              "pokemon": "Machamp",
              "item": "Adrenaline Orb",
              "moves": [
                "Dynamic Punch",
                "Stone Edge",
                "Poison Jab",
                "Knock Off"
              ],
              "ability": "No Guard"
            }
          ]
        }
      ]
    },
    "Route 105 (Optionals)": {
      "zone_name": "Route 105 (Optionals)",
      "zone_trainers": [
        {
          "trainer": "Bird Keeper Josue [Double Battle With Ruin Maniac Andres]",
          "pokemon_list": [
            {
              "pokemon": "Hawlucha",
              "item": "Focus Sash",
              "moves": [
                "Close Combat",
                "Dual Wingbeat",
                "Reversal",
                "Endeavor"
              ],
              "ability": "Mold Breaker"
            },
            {
              "pokemon": "Gliscor",
              "item": "Yache Berry",
              "moves": [
                "High Horsepower",
                "Dual Wingbeat",
                "Rock Slide",
                "Swords Dance"
              ],
              "ability": "Hyper Cutter"
            },
            {
              "pokemon": "Beedrill",
              "item": "Beedrillite",
              "moves": [
                "Poison Jab",
                "X Scissor",
                "Drill Run",
                "Protect"
              ],
              "ability": "Swarm"
            },
            {
              "pokemon": "Aurorus",
              "item": "Power Herb",
              "moves": [
                "Meteor Beam",
                "Hyper Voice",
                "Freeze Dry",
                "Icy Wind"
              ],
              "ability": "Refrigerate"
            },
            {
              "pokemon": "Wyrdeer",
              "item": "Light Clay",
              "moves": [
                "Psychic",
                "Hyper Voice",
                "Light Screen",
                "Reflect"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Gastrodon",
              "item": "Iapapa Berry",
              "moves": [
                "Earth Power",
                "Muddy Water",
                "Ice Beam",
                "Recover"
              ],
              "ability": "Storm Drain"
            }
          ]
        }
      ]
    },
    "Route 106 (Optionals)": {
      "zone_name": "Route 106 (Optionals)",
      "zone_trainers": []
    },
    "Route 107 (Optionals)": {
      "zone_name": "Route 107 (Optionals)",
      "zone_trainers": []
    },
    "Route 108 (Optionals)": {
      "zone_name": "Route 108 (Optionals)",
      "zone_trainers": [
        {
          "trainer": "Cool Trainer Carolina [Double Battle With Sailor Cory]",
          "pokemon_list": [
            {
              "pokemon": "Lopunny",
              "item": "Lopunnite",
              "moves": [
                "Return",
                "Facade",
                "Fake Out",
                "Close Combat"
              ],
              "ability": "Cute Charm"
            },
            {
              "pokemon": "Tsareena",
              "item": "Focus Sash",
              "moves": [
                "Power Whip",
                "Zen Headbutt",
                "Low Kick",
                "Flail"
              ],
              "ability": "Queenly Majesty"
            },
            {
              "pokemon": "Gyarados",
              "item": "Wacan Berry",
              "moves": [
                "Waterfall",
                "Power Whip",
                "Ice Fang",
                "Dragon Dance"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Milotic",
              "item": "Leftovers",
              "moves": [
                "Hydro Pump",
                "Muddy Water",
                "Ice Beam",
                "Hypnosis"
              ],
              "ability": "Competitive"
            },
            {
              "pokemon": "Avalugg_Hisuian",
              "item": "Rock Gem",
              "moves": [
                "Avalanche",
                "Rock Slide",
                "High Horsepower",
                "Protect"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Overqwil",
              "item": "Assault Vest",
              "moves": [
                "Gunk Shot",
                "Throat Chop",
                "Liquidation",
                "Icy Wind"
              ],
              "ability": "Intimidate"
            }
          ]
        }
      ]
    },
    "Abandoned Ship (Optionals)": {
      "zone_name": "Abandoned Ship (Optionals)",
      "zone_trainers": [
        {
          "trainer": "Ruin Maniac Garrison [Double Battle With Tuber Jani]",
          "pokemon_list": [
            {
              "pokemon": "Beheeyem",
              "item": "Power Herb",
              "moves": [
                "Psychic",
                "Meteor Beam",
                "Shadow Ball",
                "Trick Room"
              ],
              "ability": "Synchronize"
            },
            {
              "pokemon": "Trapinch",
              "item": "Ground Gem",
              "moves": [
                "High Horsepower",
                "First Impression",
                "Quick Attack",
                "Protect"
              ],
              "ability": "Arena Trap"
            },
            {
              "pokemon": "Mimikyu",
              "item": "Life Orb",
              "moves": [
                "Play Rough",
                "Shadow Claw",
                "Destiny Bond",
                "Trick Room"
              ],
              "ability": "Disguise"
            },
            {
              "pokemon": "Audino",
              "item": "Audinite",
              "moves": [
                "Moonblast",
                "Follow Me",
                "Life Dew",
                "Trick Room"
              ],
              "ability": "Regenerator"
            },
            {
              "pokemon": "Gallade",
              "item": "Room Service",
              "moves": [
                "Close Combat",
                "Zen Headbutt",
                "Bulk Up",
                "Trick Room"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Gardevoir",
              "item": "Room Service",
              "moves": [
                "Moonblast",
                "Psychic",
                "Calm Mind",
                "Trick Room"
              ],
              "ability": "Synchronize"
            }
          ]
        }
      ]
    },
    "Route 109 (Optionals)": {
      "zone_name": "Route 109 (Optionals)",
      "zone_trainers": []
    },
    "Route 118": {
      "zone_name": "Route 118",
      "zone_trainers": [
        {
          "trainer": "Swimmer\u2640 Rose [Double Battle With Swimmer\u2642 Deandre]",
          "pokemon_list": [
            {
              "pokemon": "Seaking",
              "item": "Focus Sash",
              "moves": [
                "Waterfall",
                "Megahorn",
                "Icy Wind",
                "Acupressure"
              ],
              "ability": "Lightning Rod"
            },
            {
              "pokemon": "Dewgong",
              "item": "Sitrus Berry",
              "moves": [
                "Hydro Pump",
                "Freeze Dry",
                "Drill Run",
                "Fake Out"
              ],
              "ability": "Thick Fat"
            },
            {
              "pokemon": "Whiscash",
              "item": "Lum Berry",
              "moves": [
                "Aqua Tail",
                "Stomping Tantrum",
                "Zen Headbutt",
                "Dragon Dance"
              ],
              "ability": "Oblivious"
            },
            {
              "pokemon": "Gyarados",
              "item": "Flying Gem",
              "moves": [
                "Hurricane",
                "Waterfall",
                "Crunch",
                "Ice Fang"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Samurott",
              "item": "Bug Gem",
              "moves": [
                "Liquidation",
                "Aqua Jet",
                "Megahorn",
                "Retaliate"
              ],
              "ability": "Torrent"
            }
          ]
        },
        {
          "trainer": "Fisherman Wade",
          "pokemon_list": [
            {
              "pokemon": "Starmie",
              "item": "Light Clay",
              "moves": [
                "Hydro Pump",
                "Psychic",
                "Light Screen",
                "Reflect"
              ],
              "ability": "Analytic"
            },
            {
              "pokemon": "Gorebyss",
              "item": "Rindo Berry",
              "moves": [
                "Surf",
                "Ice Beam",
                "Psychic",
                "Shell Smash"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Huntail",
              "item": "Wacan Berry",
              "moves": [
                "Aqua Tail",
                "Crunch",
                "Ice Fang",
                "Shell Smash"
              ],
              "ability": "Water Veil"
            }
          ]
        },
        {
          "trainer": "Fisherman Barny",
          "pokemon_list": [
            {
              "pokemon": "Pelipper",
              "item": "Focus Sash",
              "moves": [
                "Hurricane",
                "U Turn",
                "Knock Off",
                "Weather Ball"
              ],
              "ability": "Drizzle"
            },
            {
              "pokemon": "Basculin",
              "item": "Choice Band",
              "moves": [
                "Liquidation",
                "Flip Turn"
              ],
              "ability": "Adaptability"
            },
            {
              "pokemon": "Beartic",
              "item": "Lum Berry",
              "moves": [
                "Icicle Crash",
                "Liquidation",
                "Low Kick",
                "Swords Dance"
              ],
              "ability": "Swift Swim"
            }
          ]
        },
        {
          "trainer": "Guitarist Dalton",
          "pokemon_list": [
            {
              "pokemon": "Rillaboom",
              "item": "Choice Band",
              "moves": [
                "Wood Hammer",
                "Grassy Glide",
                "U Turn"
              ],
              "ability": "Grassy Surge"
            },
            {
              "pokemon": "Toxtricity",
              "item": "Black Sludge",
              "moves": [
                "Overdrive",
                "Nuzzle",
                "Hidden Power Grass",
                "Substitute"
              ],
              "ability": "Punk Rock"
            },
            {
              "pokemon": "Swellow",
              "item": "Toxic Orb",
              "moves": [
                "Brave Bird",
                "Facade",
                "Endeavor",
                "Protect"
              ],
              "ability": "Guts"
            }
          ]
        },
        {
          "trainer": "Bird Keeper Perry [Double Battle With Bird Keeper Chester]",
          "pokemon_list": [
            {
              "pokemon": "Vikavolt",
              "item": "Charti Berry",
              "moves": [
                "Thunderbolt",
                "Bug Buzz",
                "Electroweb",
                "Energy Ball"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Flygon",
              "item": "Yache Berry",
              "moves": [
                "Earthquake",
                "Dragon Claw",
                "Dragon Dance",
                "Roost"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Fearow",
              "item": "Scope Lens",
              "moves": [
                "Double Edge",
                "Brave Bird",
                "Drill Run",
                "Focus Energy"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Honchkrow",
              "item": "Scope Lens",
              "moves": [
                "Brave Bird",
                "Night Slash",
                "Sucker Punch",
                "Superpower"
              ],
              "ability": "Super Luck"
            }
          ]
        }
      ]
    },
    "Route 119 (West), permanent Rain": {
      "zone_name": "Route 119 (West), permanent Rain",
      "zone_trainers": [
        {
          "trainer": "Bug Maniac Taylor",
          "pokemon_list": [
            {
              "pokemon": "Escavalier",
              "item": "Assault Vest",
              "moves": [
                "Megahorn",
                "Close Combat",
                "Razor Shell",
                "Metal Burst"
              ],
              "ability": "Shell Armor"
            },
            {
              "pokemon": "Parasect",
              "item": "Bright Powder",
              "moves": [
                "Leech Life",
                "Knock Off",
                "Double Team",
                "Spore"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Shiinotic",
              "item": "Kebia Berry",
              "moves": [
                "Moonblast",
                "Pollen Puff",
                "Spore",
                "Strength Sap"
              ],
              "ability": "Rain Dish"
            },
            {
              "pokemon": "Kabutops",
              "item": "Bug Gem",
              "moves": [
                "Stone Edge",
                "Liquidation",
                "Aqua Jet",
                "Leech Life"
              ],
              "ability": "Swift Swim"
            }
          ]
        },
        {
          "trainer": "Bug Catcher Doug",
          "pokemon_list": [
            {
              "pokemon": "Shuckle",
              "item": "Custap Berry",
              "moves": [
                "Rock Tomb",
                "Final Gambit",
                "Sticky Web"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Mothim",
              "item": "Flying Gem",
              "moves": [
                "Bug Buzz",
                "Acrobatics",
                "Giga Drain",
                "Infestation"
              ],
              "ability": "Tinted Lens"
            },
            {
              "pokemon": "Illumise",
              "item": "Choice Specs",
              "moves": [
                "Bug Buzz"
              ],
              "ability": "Tinted Lens"
            }
          ]
        },
        {
          "trainer": "Fisherman Phil",
          "pokemon_list": [
            {
              "pokemon": "Luvdisc",
              "item": "Mystic Water",
              "moves": [
                "Hydro Pump",
                "Sweet Kiss"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Luvdisc",
              "item": "Mystic Water",
              "moves": [
                "Hydro Pump",
                "Sweet Kiss"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Luvdisc",
              "item": "Mystic Water",
              "moves": [
                "Hydro Pump"
              ],
              "ability": "Swift Swim"
            }
          ]
        },
        {
          "trainer": "Pokemon Ranger Lydian",
          "pokemon_list": [
            {
              "pokemon": "Kingdra",
              "item": "Lum Berry",
              "moves": [
                "Liquidation",
                "Dragon Pulse",
                "Octazooka",
                "Hurricane"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Milotic",
              "item": "Leftovers",
              "moves": [
                "Scald",
                "Flip Turn",
                "Hypnosis",
                "Recover"
              ],
              "ability": "Marvel Scale"
            },
            {
              "pokemon": "Heliolisk",
              "item": "Expert Belt",
              "moves": [
                "Thunder",
                "Dragon Pulse",
                "Weather Ball",
                "Grass Knot"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Dragalge",
              "item": "White Herb",
              "moves": [
                "Draco Meteor",
                "Sludge Wave",
                "Hydro Pump",
                "Flip Turn"
              ],
              "ability": "Adaptability"
            }
          ]
        },
        {
          "trainer": "Bug Catcher Greg",
          "pokemon_list": [
            {
              "pokemon": "Galvantula",
              "item": "Bug Gem",
              "moves": [
                "Thunder",
                "Bug Buzz",
                "Disable",
                "Sticky Web"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Frosmoth",
              "item": "Leftovers",
              "moves": [
                "Ice Beam",
                "Hurricane",
                "Weather Ball",
                "Substitute"
              ],
              "ability": "Ice Scales"
            },
            {
              "pokemon": "Beedrill",
              "item": "Scope Lens",
              "moves": [
                "X Scissor",
                "Drill Run",
                "Knock Off",
                "Focus Energy"
              ],
              "ability": "Sniper"
            }
          ]
        },
        {
          "trainer": "Bug Maniac Brent",
          "pokemon_list": [
            {
              "pokemon": "Wormadam",
              "item": "White Herb",
              "moves": [
                "Leaf Storm",
                "Bug Buzz",
                "Psychic",
                "Giga Drain"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Wormadam_Trash_Cloak",
              "item": "Leftovers",
              "moves": [
                "Iron Head",
                "Infestation",
                "Protect",
                "Metal Burst"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Wormadam_Sandy_Cloak",
              "item": "Rock Gem",
              "moves": [
                "Earthquake",
                "Rock Blast",
                "Infestation",
                "Fissure"
              ],
              "ability": "Overcoat"
            }
          ]
        },
        {
          "trainer": "Expert Donald",
          "pokemon_list": [
            {
              "pokemon": "Ludicolo",
              "item": "Focus Sash",
              "moves": [
                "Energy Ball",
                "Weather Ball",
                "Flail",
                "Counter"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Turtonator",
              "item": "Air Balloon",
              "moves": [
                "Dragon Claw",
                "Explosion",
                "Heavy Slam",
                "Curse"
              ],
              "ability": "Shell Armor"
            },
            {
              "pokemon": "Sceptile",
              "item": "Dragon Gem",
              "moves": [
                "Leaf Blade",
                "Earthquake",
                "Dual Chop",
                "Swords Dance"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Excadrill",
              "item": "Scope Lens",
              "moves": [
                "High Horsepower",
                "Iron Head",
                "Rock Slide",
                "Rock Polish"
              ],
              "ability": "Mold Breaker"
            }
          ]
        },
        {
          "trainer": "Pokemon Ranger Catherine",
          "pokemon_list": [
            {
              "pokemon": "Mr_Mime_Galarian",
              "item": "Light Clay",
              "moves": [
                "Freeze Dry",
                "Hypnosis",
                "Light Screen",
                "Reflect"
              ],
              "ability": "Vital Spirit"
            },
            {
              "pokemon": "Linoone",
              "item": "Sitrus Berry",
              "moves": [
                "Extreme Speed",
                "Stomping Tantrum",
                "Throat Chop",
                "Belly Drum"
              ],
              "ability": "Quick Feet"
            },
            {
              "pokemon": "Decidueye",
              "item": "Dark Gem",
              "moves": [
                "Poltergeist",
                "Leaf Blade",
                "Sucker Punch",
                "Swords Dance"
              ],
              "ability": "Overgrow"
            },
            {
              "pokemon": "Sylveon",
              "item": "Water Gem",
              "moves": [
                "Hyper Voice",
                "Psyshock",
                "Weather Ball",
                "Quick Attack"
              ],
              "ability": "Pixilate"
            }
          ]
        },
        {
          "trainer": "Pokemon Ranger Jackson",
          "pokemon_list": [
            {
              "pokemon": "Morpeko",
              "item": "Shuca Berry",
              "moves": [
                "Aura Wheel",
                "Seed Bomb",
                "Stomping Tantrum",
                "Super Fang"
              ],
              "ability": "Hunger Switch"
            },
            {
              "pokemon": "Ambipom",
              "item": "Silk Scarf",
              "moves": [
                "Tail Slap",
                "Low Kick",
                "Covet",
                "Fake Out"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Kingler",
              "item": "Assault Vest",
              "moves": [
                "Crabhammer",
                "High Horsepower",
                "X Scissor",
                "Knock Off"
              ],
              "ability": "Hyper Cutter"
            }
          ]
        },
        {
          "trainer": "Bug Catcher Kent",
          "pokemon_list": [
            {
              "pokemon": "Beautifly",
              "item": "Focus Sash",
              "moves": [
                "Bug Buzz",
                "Air Slash",
                "Quiver Dance",
                "Stun Spore"
              ],
              "ability": "Swarm"
            },
            {
              "pokemon": "Araquanid",
              "item": "Leftovers",
              "moves": [
                "Scald",
                "Iron Defense",
                "Protect",
                "Mirror Coat"
              ],
              "ability": "Water Bubble"
            },
            {
              "pokemon": "Scizor",
              "item": "Steel Gem",
              "moves": [
                "Bullet Punch",
                "Dual Wingbeat",
                "Roost",
                "Swords Dance"
              ],
              "ability": "Technician"
            }
          ]
        },
        {
          "trainer": "Ninja Boy Takashi [Double Battle With Psychic Dayton]",
          "pokemon_list": [
            {
              "pokemon": "Shiftry",
              "item": "Silk Scarf",
              "moves": [
                "Explosion",
                "Fake Out"
              ],
              "ability": "Early Bird"
            },
            {
              "pokemon": "Wailord",
              "item": "Life Orb",
              "moves": [
                "Water Spout",
                "Surf",
                "Explosion"
              ],
              "ability": "Oblivious"
            },
            {
              "pokemon": "Oranguru",
              "item": "Colbur Berry",
              "moves": [
                "Psychic",
                "Focus Blast",
                "Thunder",
                "Trick Room"
              ],
              "ability": "Telepathy"
            },
            {
              "pokemon": "Gardevoir",
              "item": "Fairy Gem",
              "moves": [
                "Psychic",
                "Dazzling Gleam",
                "Energy Ball",
                "Trick Room"
              ],
              "ability": "Telepathy"
            }
          ]
        },
        {
          "trainer": "Bird Keeper Hugh",
          "pokemon_list": [
            {
              "pokemon": "Skarmory",
              "item": "Custap Berry",
              "moves": [
                "Brave Bird",
                "Whirlwind",
                "Spikes",
                "Stealth Rock"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Mantine",
              "item": "Sharp Beak",
              "moves": [
                "Hurricane",
                "Scald",
                "Hidden Power Grass",
                "Roost"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Flapple",
              "item": "Liechi Berry",
              "moves": [
                "Dragon Rush",
                "Grav Apple",
                "Dragon Dance",
                "Substitute"
              ],
              "ability": "Ripen"
            },
            {
              "pokemon": "Pikachu",
              "item": "Light Ball",
              "moves": [
                "Thunder",
                "Fly",
                "Surf",
                "Knock Off"
              ],
              "ability": "Lightning Rod"
            }
          ]
        },
        {
          "trainer": "Parasol Lady Koko",
          "pokemon_list": [
            {
              "pokemon": "Barraskewda",
              "item": "Mystic Water",
              "moves": [
                "Liquidation",
                "Flip Turn",
                "Close Combat",
                "Psychic Fangs"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Roserade",
              "item": "Black Sludge",
              "moves": [
                "Sludge Bomb",
                "Grass Knot",
                "Weather Ball",
                "Sleep Powder"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Rotom_Wash",
              "item": "Lum Berry",
              "moves": [
                "Hydro Pump",
                "Thunder",
                "Volt Switch",
                "Will O Wisp"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Jynx",
              "item": "Life Orb",
              "moves": [
                "Psychic",
                "Freeze Dry",
                "Lovely Kiss",
                "Substitute"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Goodra",
              "item": "Leftovers",
              "moves": [
                "Dragon Claw",
                "Aqua Tail",
                "Curse",
                "Rest"
              ],
              "ability": "Hydration"
            }
          ]
        }
      ]
    },
    "Weather Institute": {
      "zone_name": "Weather Institute",
      "zone_trainers": [
        {
          "trainer": "Team Aqua Grunt Jazz",
          "pokemon_list": [
            {
              "pokemon": "Barbaracle",
              "item": "Bright Powder",
              "moves": [
                "Stone Edge",
                "Liquidation",
                "Aerial Ace",
                "Stealth Rock"
              ],
              "ability": "Tough Claws"
            },
            {
              "pokemon": "Breloom",
              "item": "Bright Powder",
              "moves": [
                "Mach Punch",
                "Bullet Seed",
                "Rock Tomb",
                "Spore"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Malamar",
              "item": "Quick Claw",
              "moves": [
                "Night Slash",
                "Psycho Cut",
                "Superpower",
                "Hypnosis"
              ],
              "ability": "Contrary"
            },
            {
              "pokemon": "Scolipede",
              "item": "Kings Rock",
              "moves": [
                "Pin Missile",
                "Rock Slide",
                "Smart Strike",
                "Swords Dance"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Starmie",
              "item": "Expert Belt",
              "moves": [
                "Psychic",
                "Surf",
                "Thunderbolt",
                "Ice Beam"
              ],
              "ability": "Analytic"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Drampa",
              "item": "Focus Band",
              "moves": [
                "Dragon Pulse",
                "Hurricane",
                "Surf",
                "Roost"
              ],
              "ability": "Berserk"
            },
            {
              "pokemon": "Ferrothorn",
              "item": "Quick Claw",
              "moves": [
                "Power Whip",
                "Iron Head",
                "Explosion",
                "Bulldoze"
              ],
              "ability": "Iron Barbs"
            },
            {
              "pokemon": "Exeggutor",
              "item": "Quick Claw",
              "moves": [
                "Psychic",
                "Energy Ball",
                "Explosion",
                "Low Kick"
              ],
              "ability": "Chlorophyll"
            },
            {
              "pokemon": "Omastar",
              "item": "Bright Powder",
              "moves": [
                "Surf",
                "Power Gem",
                "Earth Power",
                "Rain Dance"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Walrein",
              "item": "Leftovers",
              "moves": [
                "Liquidation",
                "Freeze Dry",
                "Body Press",
                "Curse"
              ],
              "ability": "Oblivious"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Porygon_Z",
              "item": "Chople Berry",
              "moves": [
                "Tri Attack",
                "Ice Beam",
                "Dark Pulse",
                "Hidden Power Fire"
              ],
              "ability": "Download"
            },
            {
              "pokemon": "Slowbro_Galarian",
              "item": "Quick Claw",
              "moves": [
                "Shell Side Arm",
                "Flamethrower",
                "Scald",
                "Slack Off"
              ],
              "ability": "Quick Draw"
            },
            {
              "pokemon": "Haxorus",
              "item": "Water Gem",
              "moves": [
                "Dual Chop",
                "Aqua Tail",
                "Iron Head",
                "Dragon Dance"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Weavile",
              "item": "Dark Gem",
              "moves": [
                "Knock Off",
                "Pursuit",
                "Triple Axel",
                "Psycho Cut"
              ],
              "ability": "Inner Focus"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt [Double Battle With Team Aqua Grunt]",
          "pokemon_list": [
            {
              "pokemon": "Ninetales_Alolan",
              "item": "Ice Gem",
              "moves": [
                "Moonblast",
                "Freeze Dry",
                "Hypnosis",
                "Aurora Veil"
              ],
              "ability": "Snow Warning"
            },
            {
              "pokemon": "Liepard",
              "item": "Psychic Gem",
              "moves": [
                "Assurance",
                "Psycho Cut",
                "Fake Out",
                "Fake Tears"
              ],
              "ability": "Prankster"
            },
            {
              "pokemon": "Arctozolt",
              "item": "Lum Berry",
              "moves": [
                "Blizzard",
                "Bolt Beak",
                "Icicle Crash",
                "Low Kick"
              ],
              "ability": "Slush Rush"
            },
            {
              "pokemon": "Arctovish",
              "item": "Lum Berry",
              "moves": [
                "Blizzard",
                "Hydro Pump",
                "Freeze Dry",
                "Ancient Power"
              ],
              "ability": "Slush Rush"
            }
          ]
        },
        {
          "trainer": "Aqua Admin Shelly [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Mienshao",
              "item": "Focus Sash",
              "moves": [
                "Close Combat",
                "Reversal",
                "Knock Off",
                "Fake Out"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Nihilego",
              "item": "Power Herb",
              "moves": [
                "Meteor Beam",
                "Sludge Wave",
                "Power Gem",
                "Stealth Rock"
              ],
              "ability": "Beast Boost"
            },
            {
              "pokemon": "Dragonite",
              "item": "Choice Band",
              "moves": [
                "Outrage",
                "Dual Wingbeat",
                "Aqua Tail",
                "Fire Punch"
              ],
              "ability": "Multiscale"
            },
            {
              "pokemon": "Tornadus",
              "item": "Flying Gem",
              "moves": [
                "Acrobatics",
                "Knock Off",
                "Heat Wave",
                "Grass Knot"
              ],
              "ability": "Defiant"
            },
            {
              "pokemon": "Lanturn",
              "item": "Assault Vest",
              "moves": [
                "Thunderbolt",
                "Scald",
                "Volt Switch",
                "Ice Beam"
              ],
              "ability": "Volt Absorb"
            },
            {
              "pokemon": "Blastoise",
              "item": "Blastoisinite",
              "moves": [
                "Surf",
                "Dragon Pulse",
                "Dark Pulse",
                "Shell Smash"
              ],
              "ability": "Torrent"
            }
          ]
        }
      ]
    },
    "Route 119 (East), permanent Rain and Electric Terrain": {
      "zone_name": "Route 119 (East), permanent Rain and Electric Terrain",
      "zone_trainers": [
        {
          "trainer": "Pokemon Trainer May [Double] [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Sceptile",
              "item": "Sceptilite",
              "moves": [
                "Energy Ball",
                "Dragon Pulse",
                "Earthquake",
                "Aura Sphere"
              ],
              "ability": "Overgrow"
            },
            {
              "pokemon": "Mantine",
              "item": "Life Orb",
              "moves": [
                "Hydro Pump",
                "Hurricane",
                "Surf",
                "Helping Hand"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Goodra",
              "item": "Leftovers",
              "moves": [
                "Dragon Pulse",
                "Thunder",
                "Muddy Water",
                "Rest"
              ],
              "ability": "Hydration"
            },
            {
              "pokemon": "Raichu_Alolan",
              "item": "Focus Sash",
              "moves": [
                "Thunder",
                "Psychic",
                "Endeavor",
                "Fake Out"
              ],
              "ability": "Surge Surfer"
            },
            {
              "pokemon": "Lucario",
              "item": "Fighting Gem",
              "moves": [
                "Close Combat",
                "Flash Cannon",
                "Vacuum Wave",
                "Terrain Pulse"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Gastrodon",
              "item": "Rindo Berry",
              "moves": [
                "Earth Power",
                "Muddy Water",
                "Icy Wind",
                "Recover"
              ],
              "ability": "Storm Drain"
            }
          ]
        },
        {
          "trainer": "Pokemon Trainer May [Double] [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Blaziken",
              "item": "Blazikenite",
              "moves": [
                "Close Combat",
                "Brave Bird",
                "Thunder Punch",
                "Swords Dance"
              ],
              "ability": "Blaze"
            },
            {
              "pokemon": "Ludicolo",
              "item": "Life Orb",
              "moves": [
                "Energy Ball",
                "Zen Headbutt",
                "Weather Ball",
                "Fake Out"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Salamence",
              "item": "Flying Gem",
              "moves": [
                "Hurricane",
                "Dragon Pulse",
                "Hydro Pump",
                "Roost"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Heliolisk",
              "item": "Focus Sash",
              "moves": [
                "Thunder",
                "Hyper Voice",
                "Weather Ball",
                "Grass Knot"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Metagross",
              "item": "Assault Vest",
              "moves": [
                "Meteor Mash",
                "Zen Headbutt",
                "Bullet Punch",
                "Thunder Punch"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Blastoise",
              "item": "Leftovers",
              "moves": [
                "Muddy Water",
                "Terrain Pulse",
                "Fake Out",
                "Protect"
              ],
              "ability": "Rain Dish"
            }
          ]
        },
        {
          "trainer": "Pokemon Trainer May [Double] [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Swampert",
              "item": "Swampertite",
              "moves": [
                "Earthquake",
                "High Horsepower",
                "Liquidation",
                "Stone Edge"
              ],
              "ability": "Torrent"
            },
            {
              "pokemon": "Gyarados",
              "item": "Life Orb",
              "moves": [
                "Hurricane",
                "Muddy Water",
                "Waterfall",
                "Thunder"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Dragapult",
              "item": "Dragon Gem",
              "moves": [
                "Shadow Ball",
                "Breaking Swipe",
                "Hydro Pump",
                "Thunder"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Magnezone",
              "item": "Custap Berry",
              "moves": [
                "Thunder",
                "Flash Cannon",
                "Electroweb",
                "Body Press"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Toxicroak",
              "item": "Focus Sash",
              "moves": [
                "Gunk Shot",
                "Cross Chop",
                "Vacuum Wave",
                "Fake Out"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Exeggutor_Alolan",
              "item": "Iapapa Berry",
              "moves": [
                "Wood Hammer",
                "Dragon Hammer",
                "Low Kick",
                "Terrain Pulse"
              ],
              "ability": "Harvest"
            }
          ]
        },
        {
          "trainer": "Ninja Boy Yasu [Double Battle With Guitarist Fabian]",
          "pokemon_list": [
            {
              "pokemon": "Banette",
              "item": "Focus Sash",
              "moves": [
                "Poltergeist",
                "Shadow Sneak",
                "Icy Wind",
                "Will O Wisp"
              ],
              "ability": "Cursed Body"
            },
            {
              "pokemon": "Castform",
              "item": "Focus Sash",
              "moves": [
                "Thunder",
                "Hurricane",
                "Weather Ball",
                "Icy Wind"
              ],
              "ability": "Forecast"
            },
            {
              "pokemon": "Jolteon",
              "item": "Life Orb",
              "moves": [
                "Thunder",
                "Hyper Voice",
                "Weather Ball",
                "Detect"
              ],
              "ability": "Volt Absorb"
            },
            {
              "pokemon": "Obstagoon",
              "item": "Normal Gem",
              "moves": [
                "Frustration",
                "Throat Chop",
                "Retaliate",
                "Close Combat"
              ],
              "ability": "Defiant"
            }
          ]
        },
        {
          "trainer": "Ninja Boy Gren",
          "pokemon_list": [
            {
              "pokemon": "Forretress",
              "item": "Custap Berry",
              "moves": [
                "Gyro Ball",
                "Explosion",
                "Spikes",
                "Toxic Spikes"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Dusknoir",
              "item": "Red Card",
              "moves": [
                "Poltergeist",
                "Shadow Sneak",
                "Thunder Punch",
                "Revenge"
              ],
              "ability": "Pressure"
            },
            {
              "pokemon": "Ditto",
              "item": "Quick Claw",
              "moves": [
                "Transform"
              ],
              "ability": "Imposter"
            },
            {
              "pokemon": "Togedemaru",
              "item": "Weakness Policy",
              "moves": [
                "Steel Roller",
                "Iron Head",
                "Zing Zap",
                "Reversal"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Greninja_Battle_Bond",
              "item": "Mystic Water",
              "moves": [
                "Surf",
                "Dark Pulse",
                "Water Shuriken",
                "Extrasensory"
              ],
              "ability": "Battle Bond"
            }
          ]
        }
      ]
    },
    "Route 120 (North)": {
      "zone_name": "Route 120 (North)",
      "zone_trainers": [
        {
          "trainer": "Parasol Lady Clarissa",
          "pokemon_list": [
            {
              "pokemon": "Greedent",
              "item": "Custap Berry",
              "moves": [
                "Body Slam",
                "Earthquake",
                "Crunch",
                "Swords Dance"
              ],
              "ability": "Gluttony"
            },
            {
              "pokemon": "Amoonguss",
              "item": "Black Sludge",
              "moves": [
                "Energy Ball",
                "Sludge Bomb",
                "Foul Play",
                "Spore"
              ],
              "ability": "Effect Spore"
            },
            {
              "pokemon": "Hatterene",
              "item": "Leftovers",
              "moves": [
                "Psychic",
                "Draining Kiss",
                "Calm Mind",
                "Protect"
              ],
              "ability": "Magic Bounce"
            }
          ]
        },
        {
          "trainer": "Bird Keeper Robert",
          "pokemon_list": [
            {
              "pokemon": "Vikavolt",
              "item": "Power Herb",
              "moves": [
                "Bug Buzz",
                "Thunderbolt",
                "Solar Beam",
                "Guillotine"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Tropius",
              "item": "Sitrus Berry",
              "moves": [
                "Air Slash",
                "Earthquake",
                "Leech Seed",
                "Substitute"
              ],
              "ability": "Harvest"
            },
            {
              "pokemon": "Delibird",
              "item": "Choice Band",
              "moves": [
                "Dual Wingbeat",
                "Triple Axel"
              ],
              "ability": "Hustle"
            }
          ]
        }
      ]
    },
    "Fortree Gym": {
      "zone_name": "Fortree Gym",
      "zone_trainers": [
        {
          "trainer": "Expert Kevin",
          "pokemon_list": [
            {
              "pokemon": "Masquerain",
              "item": "Focus Sash",
              "moves": [
                "Bug Buzz",
                "Hydro Pump",
                "Sticky Web",
                "Stun Spore"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Rotom_Mow",
              "item": "White Herb",
              "moves": [
                "Leaf Storm",
                "Discharge",
                "Hidden Power Ground",
                "Thunder Wave"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Porygon2",
              "item": "Eviolite",
              "moves": [
                "Tri Attack",
                "Ice Beam",
                "Recover",
                "Thunder Wave"
              ],
              "ability": "Download"
            },
            {
              "pokemon": "Gyarados",
              "item": "Wacan Berry",
              "moves": [
                "Waterfall",
                "Earthquake",
                "Dragon Dance",
                "Thunder Wave"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Togekiss",
              "item": "Leftovers",
              "moves": [
                "Air Slash",
                "Flamethrower",
                "Aura Sphere",
                "Nasty Plot"
              ],
              "ability": "Serene Grace"
            }
          ]
        },
        {
          "trainer": "Picnicker Ashley",
          "pokemon_list": [
            {
              "pokemon": "Pelipper",
              "item": "Choice Scarf",
              "moves": [
                "Hurricane",
                "Weather Ball"
              ],
              "ability": "Drizzle"
            },
            {
              "pokemon": "Decidueye",
              "item": "Starf Berry",
              "moves": [
                "Leaf Blade",
                "Shadow Ball",
                "Brave Bird",
                "Substitute"
              ],
              "ability": "Overgrow"
            },
            {
              "pokemon": "Scizor",
              "item": "Flying Gem",
              "moves": [
                "Bug Bite",
                "Bullet Punch",
                "Acrobatics",
                "Swords Dance"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Golduck",
              "item": "Psychic Gem",
              "moves": [
                "Surf",
                "Psyshock",
                "Ice Beam",
                "Hypnosis"
              ],
              "ability": "Swift Swim"
            }
          ]
        },
        {
          "trainer": "Camper Flint [Double Battle With Bird Keeper Edwardo]",
          "pokemon_list": [
            {
              "pokemon": "Ribombee",
              "item": "Focus Sash",
              "moves": [
                "Pollen Puff",
                "Dazzling Gleam",
                "Energy Ball",
                "Quiver Dance"
              ],
              "ability": "Shield Dust"
            },
            {
              "pokemon": "Venomoth",
              "item": "Lum Berry",
              "moves": [
                "Bug Buzz",
                "Sludge Bomb",
                "Quiver Dance",
                "Sleep Powder"
              ],
              "ability": "Wonder Skin"
            },
            {
              "pokemon": "Oricorio_Pom_Pom",
              "item": "Iapapa Berry",
              "moves": [
                "Revelation Dance",
                "Air Slash",
                "Icy Wind",
                "Feather Dance"
              ],
              "ability": "Dancer"
            },
            {
              "pokemon": "Oricorio",
              "item": "Fire Gem",
              "moves": [
                "Revelation Dance",
                "Air Slash",
                "Teeter Dance",
                "Roost"
              ],
              "ability": "Dancer"
            },
            {
              "pokemon": "Oricorio_Pau",
              "item": "Iapapa Berry",
              "moves": [
                "Revelation Dance",
                "Hidden Power Fire",
                "Teeter Dance",
                "Calm Mind"
              ],
              "ability": "Dancer"
            }
          ]
        },
        {
          "trainer": "Bird Keeper Darius",
          "pokemon_list": [
            {
              "pokemon": "Clefable",
              "item": "Life Orb",
              "moves": [
                "Moonblast",
                "Flamethrower",
                "Psychic",
                "Stealth Rock"
              ],
              "ability": "Magic Guard"
            },
            {
              "pokemon": "Noivern",
              "item": "Flying Gem",
              "moves": [
                "Hurricane",
                "Dragon Pulse",
                "Flamethrower",
                "Super Fang"
              ],
              "ability": "Infiltrator"
            },
            {
              "pokemon": "Crobat",
              "item": "Scope Lens",
              "moves": [
                "Brave Bird",
                "Cross Poison",
                "Heat Wave",
                "Hypnosis"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Gliscor",
              "item": "Yache Berry",
              "moves": [
                "Earthquake",
                "Dual Wingbeat",
                "Sky Uppercut",
                "Swords Dance"
              ],
              "ability": "Hyper Cutter"
            }
          ]
        },
        {
          "trainer": "Bird Keeper Jared",
          "pokemon_list": [
            {
              "pokemon": "Yanmega",
              "item": "Choice Scarf",
              "moves": [
                "Bug Buzz",
                "Air Slash",
                "U Turn"
              ],
              "ability": "Tinted Lens"
            },
            {
              "pokemon": "Sirfetchd",
              "item": "Lum Berry",
              "moves": [
                "Close Combat",
                "Dual Wingbeat",
                "Poison Jab",
                "Quick Attack"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Corviknight",
              "item": "Leftovers",
              "moves": [
                "Dual Wingbeat",
                "Body Press",
                "Bulk Up",
                "Roost"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Archeops",
              "item": "Focus Sash",
              "moves": [
                "Stone Edge",
                "Dual Wingbeat",
                "Endeavor",
                "Stealth Rock"
              ],
              "ability": "Defeatist"
            },
            {
              "pokemon": "Blaziken",
              "item": "Flying Gem",
              "moves": [
                "Close Combat",
                "Flare Blitz",
                "Acrobatics",
                "Swords Dance"
              ],
              "ability": "Speed Boost"
            }
          ]
        },
        {
          "trainer": "Leader Winona [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Staraptor",
              "item": "Choice Scarf",
              "moves": [
                "Brave Bird",
                "Double Edge",
                "Close Combat",
                "U Turn"
              ],
              "ability": "Reckless"
            },
            {
              "pokemon": "Volcarona",
              "item": "Focus Sash",
              "moves": [
                "Bug Buzz",
                "Fiery Dance",
                "Psychic",
                "Quiver Dance"
              ],
              "ability": "Swarm"
            },
            {
              "pokemon": "Hawlucha",
              "item": "Flying Gem",
              "moves": [
                "Close Combat",
                "Acrobatics",
                "Stone Edge",
                "Swords Dance"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Celesteela",
              "item": "Assault Vest",
              "moves": [
                "Heavy Slam",
                "Earthquake",
                "Flamethrower",
                "Rock Slide"
              ],
              "ability": "Beast Boost"
            },
            {
              "pokemon": "Shaymin_Sky",
              "item": "Leftovers",
              "moves": [
                "Seed Flare",
                "Air Slash",
                "Earth Power",
                "Aromatherapy"
              ],
              "ability": "Serene Grace"
            },
            {
              "pokemon": "Altaria",
              "item": "Altarianite",
              "moves": [
                "Earthquake",
                "Hyper Voice",
                "Flamethrower",
                "Roost"
              ],
              "ability": "Natural Cure"
            }
          ]
        }
      ]
    },
    "split_name": "Winona"
  },
  "TnL": {
    "Route 120 (South)": {
      "zone_name": "Route 120 (South)",
      "zone_trainers": [
        {
          "trainer": "Bird Keeper Colin",
          "pokemon_list": [
            {
              "pokemon": "Aerodactyl",
              "item": "Passho Berry",
              "moves": [
                "Stone Edge",
                "Aqua Tail",
                "Substitute",
                "Swagger"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Charizard",
              "item": "Focus Sash",
              "moves": [
                "Burn Up",
                "Hurricane",
                "Weather Ball",
                "Counter"
              ],
              "ability": "Blaze"
            },
            {
              "pokemon": "Empoleon",
              "item": "Water Gem",
              "moves": [
                "Iron Head",
                "Aqua Jet",
                "Drill Peck",
                "Swords Dance"
              ],
              "ability": "Defiant"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Gian",
          "pokemon_list": [
            {
              "pokemon": "Drifblim",
              "item": "Colbur Berry",
              "moves": [
                "Air Slash",
                "Explosion",
                "Thunder",
                "Weather Ball"
              ],
              "ability": "Aftermath"
            },
            {
              "pokemon": "Swampert",
              "item": "Rindo Berry",
              "moves": [
                "Liquidation",
                "Flip Turn",
                "Earthquake",
                "Stealth Rock"
              ],
              "ability": "Torrent"
            },
            {
              "pokemon": "Duraludon",
              "item": "Leftovers",
              "moves": [
                "Draco Meteor",
                "Dragon Tail",
                "Heavy Slam",
                "Body Press"
              ],
              "ability": "Heavy Metal"
            },
            {
              "pokemon": "Alakazam",
              "item": "Focus Sash",
              "moves": [
                "Psychic",
                "Dazzling Gleam",
                "Signal Beam",
                "Counter"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Alcremie",
              "item": "Babiri Berry",
              "moves": [
                "Draining Kiss",
                "Stored Power",
                "Acid Armor",
                "Calm Mind"
              ],
              "ability": "Sweet Veil"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Jennifer [Double Battle With Battle Girl Callie]",
          "pokemon_list": [
            {
              "pokemon": "Metagross",
              "item": "Assault Vest",
              "moves": [
                "Meteor Mash",
                "Zen Headbutt",
                "Thunder Punch",
                "Icy Wind"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Ursaring",
              "item": "Toxic Orb",
              "moves": [
                "Facade",
                "Retaliate",
                "Gunk Shot",
                "High Horsepower"
              ],
              "ability": "Quick Feet"
            },
            {
              "pokemon": "Zebstrika",
              "item": "Life Orb",
              "moves": [
                "Zing Zap",
                "High Horsepower",
                "Low Kick",
                "Magnet Rise"
              ],
              "ability": "Lightning Rod"
            },
            {
              "pokemon": "Sawk",
              "item": "Lum Berry",
              "moves": [
                "Close Combat",
                "Throat Chop",
                "Coaching",
                "Protect"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Poliwrath",
              "item": "Lum Berry",
              "moves": [
                "Close Combat",
                "Hydro Pump",
                "Coaching",
                "Hypnosis"
              ],
              "ability": "Swift Swim"
            }
          ]
        },
        {
          "trainer": "Parasol Lady Angelica",
          "pokemon_list": [
            {
              "pokemon": "Stunfisk",
              "item": "Lum Berry",
              "moves": [
                "Earthquake",
                "Discharge",
                "Scald",
                "Stealth Rock"
              ],
              "ability": "Static"
            },
            {
              "pokemon": "Chansey",
              "item": "Eviolite",
              "moves": [
                "Seismic Toss",
                "Counter",
                "Soft Boiled",
                "Toxic"
              ],
              "ability": "Natural Cure"
            },
            {
              "pokemon": "Mismagius",
              "item": "Ghost Gem",
              "moves": [
                "Shadow Ball",
                "Dazzling Gleam",
                "Mystical Fire",
                "Nasty Plot"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Musharna",
              "item": "Colbur Berry",
              "moves": [
                "Psychic",
                "Moonblast",
                "Calm Mind",
                "Dark Void"
              ],
              "ability": "Synchronize"
            }
          ]
        },
        {
          "trainer": "Pokemon Ranger Jenna",
          "pokemon_list": [
            {
              "pokemon": "Dragalge",
              "item": "Shuca Berry",
              "moves": [
                "Sludge Wave",
                "Dragon Pulse",
                "Scald",
                "Toxic Spikes"
              ],
              "ability": "Adaptability"
            },
            {
              "pokemon": "Pangoro",
              "item": "Eject Pack",
              "moves": [
                "Close Combat",
                "Knock Off",
                "Gunk Shot"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Gothitelle",
              "item": "Focus Sash",
              "moves": [
                "Psychic",
                "Thunderbolt",
                "Mystical Fire",
                "Grass Knot"
              ],
              "ability": "Shadow Tag"
            },
            {
              "pokemon": "Lilligant",
              "item": "Leftovers",
              "moves": [
                "Petal Dance",
                "Sleep Powder",
                "Synthesis",
                "Quiver Dance"
              ],
              "ability": "Own Tempo"
            }
          ]
        },
        {
          "trainer": "Pokemon Ranger Lorenzo",
          "pokemon_list": [
            {
              "pokemon": "Golisopod",
              "item": "Bug Gem",
              "moves": [
                "First Impression",
                "Liquidation",
                "Leech Life",
                "Knock Off"
              ],
              "ability": "Emergency Exit"
            },
            {
              "pokemon": "Appletun",
              "item": "Lum Berry",
              "moves": [
                "Apple Acid",
                "Body Press",
                "Iron Defense",
                "Recover"
              ],
              "ability": "Thick Fat"
            },
            {
              "pokemon": "Miltank",
              "item": "White Herb",
              "moves": [
                "Body Slam",
                "Body Press",
                "Zen Headbutt",
                "Curse"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Mr_Mime",
              "item": "Life Orb",
              "moves": [
                "Moonblast",
                "Psychic",
                "Mystical Fire",
                "Grass Knot"
              ],
              "ability": "Filter"
            }
          ]
        },
        {
          "trainer": "Bug Maniac Jeffrey",
          "pokemon_list": [
            {
              "pokemon": "Vivillon_Elegant",
              "item": "Focus Sash",
              "moves": [
                "Air Slash",
                "Energy Ball",
                "Electroweb",
                "Endeavor"
              ],
              "ability": "Shield Dust"
            },
            {
              "pokemon": "Vivillon_Modern",
              "item": "Focus Sash",
              "moves": [
                "Bug Buzz",
                "Air Slash",
                "Psychic",
                "Quiver Dance"
              ],
              "ability": "Shield Dust"
            },
            {
              "pokemon": "Vivillon_Sun",
              "item": "Life Orb",
              "moves": [
                "Hurricane",
                "Bug Buzz",
                "Quiver Dance",
                "Sleep Powder"
              ],
              "ability": "Compound Eyes"
            }
          ]
        },
        {
          "trainer": "Ruin Maniac Chip",
          "pokemon_list": [
            {
              "pokemon": "Armaldo",
              "item": "Iapapa Berry",
              "moves": [
                "Leech Life",
                "Rock Tomb",
                "Knock Off",
                "Stealth Rock"
              ],
              "ability": "Battle Armor"
            },
            {
              "pokemon": "Cradily",
              "item": "Leftovers",
              "moves": [
                "Power Whip",
                "Rock Slide",
                "Curse",
                "Rest"
              ],
              "ability": "Suction Cups"
            },
            {
              "pokemon": "Tyrantrum",
              "item": "Dragon Gem",
              "moves": [
                "Head Smash",
                "Scale Shot",
                "Earthquake",
                "Substitute"
              ],
              "ability": "Rock Head"
            },
            {
              "pokemon": "Aurorus",
              "item": "Ice Gem",
              "moves": [
                "Hyper Voice",
                "Freeze Dry",
                "Earth Power",
                "Substitute"
              ],
              "ability": "Refrigerate"
            }
          ]
        },
        {
          "trainer": "Ninja Boy Keigo",
          "pokemon_list": [
            {
              "pokemon": "Lickilicky",
              "item": "Chople Berry",
              "moves": [
                "Explosion",
                "Body Slam",
                "Power Whip",
                "Knock Off"
              ],
              "ability": "Own Tempo"
            },
            {
              "pokemon": "Cursola",
              "item": "Custap Berry",
              "moves": [
                "Shadow Ball",
                "Explosion",
                "Hydro Pump",
                "Endure"
              ],
              "ability": "Perish Body"
            },
            {
              "pokemon": "Glalie",
              "item": "Leftovers",
              "moves": [
                "Freeze Dry",
                "Icy Wind",
                "Explosion",
                "Protect"
              ],
              "ability": "Moody"
            },
            {
              "pokemon": "Zoroark",
              "item": "Dark Gem",
              "moves": [
                "Night Daze",
                "Flamethrower",
                "Aura Sphere",
                "Nasty Plot"
              ],
              "ability": "Illusion"
            }
          ]
        },
        {
          "trainer": "Ninja Boy Riley",
          "pokemon_list": [
            {
              "pokemon": "Greninja",
              "item": "Life Orb",
              "moves": [
                "Hydro Pump",
                "Gunk Shot",
                "Ice Beam",
                "Extrasensory"
              ],
              "ability": "Protean"
            },
            {
              "pokemon": "Dugtrio",
              "item": "Ground Gem",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Final Gambit",
                "Memento"
              ],
              "ability": "Arena Trap"
            },
            {
              "pokemon": "Bisharp",
              "item": "Black Glasses",
              "moves": [
                "Iron Head",
                "Sucker Punch",
                "Pursuit",
                "Zen Headbutt"
              ],
              "ability": "Defiant"
            },
            {
              "pokemon": "Durant",
              "item": "Focus Sash",
              "moves": [
                "X Scissor",
                "Rock Slide",
                "Superpower",
                "Swords Dance"
              ],
              "ability": "Swarm"
            }
          ]
        }
      ]
    },
    "Route 121": {
      "zone_name": "Route 121",
      "zone_trainers": [
        {
          "trainer": "Cool Trainer Tammy [Double Battle With Bug Maniac Cale]",
          "pokemon_list": [
            {
              "pokemon": "Slaking",
              "item": "Lum Berry",
              "moves": [
                "Return",
                "Earthquake",
                "Fire Punch",
                "Knock Off"
              ],
              "ability": "Truant"
            },
            {
              "pokemon": "Archeops",
              "item": "Rock Gem",
              "moves": [
                "Head Smash",
                "Rock Slide",
                "Dual Wingbeat",
                "Earthquake"
              ],
              "ability": "Defeatist"
            },
            {
              "pokemon": "Weezing_Galarian",
              "item": "Air Balloon",
              "moves": [
                "Gunk Shot",
                "Strange Steam",
                "Assurance",
                "Protect"
              ],
              "ability": "Neutralizing Gas"
            },
            {
              "pokemon": "Weezing",
              "item": "Air Balloon",
              "moves": [
                "Sludge Bomb",
                "Fire Blast",
                "Assurance",
                "Protect"
              ],
              "ability": "Neutralizing Gas"
            }
          ]
        },
        {
          "trainer": "Beauty Jessica",
          "pokemon_list": [
            {
              "pokemon": "Froslass",
              "item": "Ice Gem",
              "moves": [
                "Blizzard",
                "Shadow Ball",
                "Freeze Dry",
                "Spikes"
              ],
              "ability": "Cursed Body"
            },
            {
              "pokemon": "Jynx",
              "item": "Expert Belt",
              "moves": [
                "Ice Beam",
                "Psychic",
                "Aura Sphere",
                "Lovely Kiss"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Florges",
              "item": "Leftovers",
              "moves": [
                "Moonblast",
                "Psychic",
                "Double Team",
                "Protect"
              ],
              "ability": "Flower Veil"
            },
            {
              "pokemon": "Kangaskhan",
              "item": "Leftovers",
              "moves": [
                "Double Edge",
                "Earthquake",
                "Drain Punch",
                "Sing"
              ],
              "ability": "Scrappy"
            }
          ]
        },
        {
          "trainer": "Pokemon Breeder Pat",
          "pokemon_list": [
            {
              "pokemon": "Sudowoodo",
              "item": "Custap Berry",
              "moves": [
                "Stone Edge",
                "Explosion",
                "Earthquake",
                "Flail"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Porygon2",
              "item": "Eviolite",
              "moves": [
                "Tri Attack",
                "Psychic",
                "Dark Pulse",
                "Agility"
              ],
              "ability": "Download"
            },
            {
              "pokemon": "Magneton",
              "item": "Eviolite",
              "moves": [
                "Thunderbolt",
                "Flash Cannon",
                "Hidden Power Grass",
                "Magnet Rise"
              ],
              "ability": "Magnet Pull"
            },
            {
              "pokemon": "Garbodor",
              "item": "Shuca Berry",
              "moves": [
                "Gunk Shot",
                "Explosion",
                "Seed Bomb",
                "Drain Punch"
              ],
              "ability": "Weak Armor"
            },
            {
              "pokemon": "Coalossal",
              "item": "Fire Gem",
              "moves": [
                "Burn Up",
                "Power Gem",
                "Earth Power",
                "Will O Wisp"
              ],
              "ability": "Steam Engine"
            }
          ]
        },
        {
          "trainer": "Pokemon Breeder Myles",
          "pokemon_list": [
            {
              "pokemon": "Pyroar",
              "item": "Power Herb",
              "moves": [
                "Fire Blast",
                "Hyper Voice",
                "Solar Beam",
                "Scorching Sands"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Oranguru",
              "item": "Colbur Berry",
              "moves": [
                "Psychic",
                "Aura Sphere",
                "Shadow Ball",
                "Calm Mind"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Passimian",
              "item": "Lum Berry",
              "moves": [
                "Close Combat",
                "Iron Tail",
                "Knock Off",
                "Rock Slide"
              ],
              "ability": "Defiant"
            },
            {
              "pokemon": "Mudsdale",
              "item": "Leftovers",
              "moves": [
                "Earthquake",
                "Body Press",
                "Smack Down",
                "Substitute"
              ],
              "ability": "Stamina"
            },
            {
              "pokemon": "Furret",
              "item": "Dark Gem",
              "moves": [
                "Double Edge",
                "Sucker Punch",
                "Baton Pass",
                "Swords Dance"
              ],
              "ability": "Keen Eye"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Gustavo",
          "pokemon_list": [
            {
              "pokemon": "Eelektross",
              "item": "Lum Berry",
              "moves": [
                "Thunderbolt",
                "Flamethrower",
                "Grass Knot",
                "Toxic"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Conkeldurr",
              "item": "Assault Vest",
              "moves": [
                "Drain Punch",
                "Mach Punch",
                "Ice Punch",
                "Counter"
              ],
              "ability": "Iron Fist"
            },
            {
              "pokemon": "Jellicent",
              "item": "Leftovers",
              "moves": [
                "Scald",
                "Hex",
                "Toxic",
                "Strength Sap"
              ],
              "ability": "Cursed Body"
            },
            {
              "pokemon": "Scrafty",
              "item": "Poison Gem",
              "moves": [
                "Close Combat",
                "Crunch",
                "Poison Jab",
                "Dragon Dance"
              ],
              "ability": "Moxie"
            },
            {
              "pokemon": "Hydreigon",
              "item": "Lum Berry",
              "moves": [
                "Dragon Pulse",
                "Dark Pulse",
                "Nasty Plot",
                "Roost"
              ],
              "ability": "Levitate"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Cristin",
          "pokemon_list": [
            {
              "pokemon": "Hippowdon",
              "item": "Lum Berry",
              "moves": [
                "Earthquake",
                "Crunch",
                "Stealth Rock",
                "Slack Off"
              ],
              "ability": "Sand Stream"
            },
            {
              "pokemon": "Tyranitar",
              "item": "Chople Berry",
              "moves": [
                "Stone Edge",
                "Pursuit",
                "Earthquake",
                "Counter"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Stoutland",
              "item": "Normal Gem",
              "moves": [
                "Return",
                "Retaliate",
                "Close Combat",
                "Play Rough"
              ],
              "ability": "Sand Rush"
            },
            {
              "pokemon": "Dracozolt",
              "item": "Dragon Gem",
              "moves": [
                "Draco Meteor",
                "Dragon Rush",
                "Bolt Beak",
                "Earthquake"
              ],
              "ability": "Sand Rush"
            }
          ]
        },
        {
          "trainer": "Young Couple Brian&Casey [Double]",
          "pokemon_list": [
            {
              "pokemon": "Audino",
              "item": "Chople Berry",
              "moves": [
                "Icy Wind",
                "Heal Pulse",
                "Helping Hand",
                "Follow Me"
              ],
              "ability": "Regenerator"
            },
            {
              "pokemon": "Gengar",
              "item": "Poison Gem",
              "moves": [
                "Sludge Wave",
                "Shadow Ball",
                "Aura Sphere",
                "Protect"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Heracross",
              "item": "Focus Sash",
              "moves": [
                "Close Combat",
                "Megahorn",
                "Rock Slide",
                "Smart Strike"
              ],
              "ability": "Moxie"
            },
            {
              "pokemon": "Mamoswine",
              "item": "Lum Berry",
              "moves": [
                "High Horsepower",
                "Icicle Crash",
                "Ice Shard",
                "Knock Off"
              ],
              "ability": "Oblivious"
            }
          ]
        },
        {
          "trainer": "Gentleman Walter",
          "pokemon_list": [
            {
              "pokemon": "Granbull",
              "item": "Iapapa Berry",
              "moves": [
                "Play Rough",
                "Low Kick",
                "Super Fang",
                "Thunder Wave"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Bewear",
              "item": "Leftovers",
              "moves": [
                "Drain Punch",
                "Payback",
                "Bulk Up",
                "Protect"
              ],
              "ability": "Fluffy"
            },
            {
              "pokemon": "Manectric",
              "item": "Life Orb",
              "moves": [
                "Thunderbolt",
                "Flamethrower",
                "Hidden Power Grass",
                "Magnet Rise"
              ],
              "ability": "Static"
            },
            {
              "pokemon": "Silvally_Dark",
              "item": "Dark Memory",
              "moves": [
                "Multi Attack",
                "Pursuit",
                "Explosion",
                "Flamethrower"
              ],
              "ability": "RKS System"
            },
            {
              "pokemon": "Azumarill",
              "item": "Water Gem",
              "moves": [
                "Aqua Tail",
                "Play Rough",
                "Aqua Jet",
                "Knock Off"
              ],
              "ability": "Huge Power"
            }
          ]
        },
        {
          "trainer": "Pok\u00e9fan Vanessa",
          "pokemon_list": [
            {
              "pokemon": "Mimikyu",
              "item": "Iron Ball",
              "moves": [
                "Play Rough",
                "Shadow Claw",
                "Swords Dance",
                "Trick Room"
              ],
              "ability": "Disguise"
            },
            {
              "pokemon": "Aromatisse",
              "item": "Fairy Gem",
              "moves": [
                "Moonblast",
                "Psychic",
                "Nasty Plot",
                "Trick Room"
              ],
              "ability": "Aroma Veil"
            },
            {
              "pokemon": "Reuniclus",
              "item": "Life Orb",
              "moves": [
                "Psychic",
                "Thunder",
                "Signal Beam",
                "Trick Room"
              ],
              "ability": "Magic Guard"
            }
          ]
        }
      ]
    },
    "Lilycove": {
      "zone_name": "Lilycove",
      "zone_trainers": [
        {
          "trainer": "Pokemon Trainer May [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Garchomp",
              "item": "Yache Berry",
              "moves": [
                "Earthquake",
                "Dual Chop",
                "Fire Blast",
                "Stealth Rock"
              ],
              "ability": "Rough Skin"
            },
            {
              "pokemon": "Machamp",
              "item": "Leftovers",
              "moves": [
                "Dynamic Punch",
                "Stone Edge",
                "Knock Off",
                "Substitute"
              ],
              "ability": "No Guard"
            },
            {
              "pokemon": "Alakazam",
              "item": "Focus Sash",
              "moves": [
                "Psychic",
                "Dazzling Gleam",
                "Shadow Ball",
                "Nasty Plot"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Melmetal",
              "item": "Assault Vest",
              "moves": [
                "Double Iron Bash",
                "Body Press",
                "Ice Punch",
                "Thunder Punch"
              ],
              "ability": "Iron Fist"
            },
            {
              "pokemon": "Moltres_Galarian",
              "item": "Iapapa Berry",
              "moves": [
                "Hurricane",
                "Fiery Wrath",
                "Heat Wave",
                "Agility"
              ],
              "ability": "Berserk"
            },
            {
              "pokemon": "Sceptile",
              "item": "Sceptilite",
              "moves": [
                "Leaf Blade",
                "Dragon Claw",
                "Earthquake",
                "Swords Dance"
              ],
              "ability": "Overgrow"
            }
          ]
        },
        {
          "trainer": "Pokemon Trainer May [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Garchomp",
              "item": "Yache Berry",
              "moves": [
                "Earthquake",
                "Dual Chop",
                "Fire Blast",
                "Stealth Rock"
              ],
              "ability": "Rough Skin"
            },
            {
              "pokemon": "Machamp",
              "item": "Leftovers",
              "moves": [
                "Dynamic Punch",
                "Stone Edge",
                "Knock Off",
                "Substitute"
              ],
              "ability": "No Guard"
            },
            {
              "pokemon": "Weavile",
              "item": "Life Orb",
              "moves": [
                "Knock Off",
                "Ice Shard",
                "Pursuit",
                "Triple Axel"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Melmetal",
              "item": "Assault Vest",
              "moves": [
                "Double Iron Bash",
                "Body Press",
                "Ice Punch",
                "Thunder Punch"
              ],
              "ability": "Iron Fist"
            },
            {
              "pokemon": "Articuno_Galarian",
              "item": "Lum Berry",
              "moves": [
                "Hurricane",
                "Freezing Glare",
                "Calm Mind",
                "Recover"
              ],
              "ability": "Competitive"
            },
            {
              "pokemon": "Blaziken",
              "item": "Blazikenite",
              "moves": [
                "Close Combat",
                "Flare Blitz",
                "Knock Off",
                "Swords Dance"
              ],
              "ability": "Blaze"
            }
          ]
        },
        {
          "trainer": "Pokemon Trainer May [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Garchomp",
              "item": "Yache Berry",
              "moves": [
                "Earthquake",
                "Dual Chop",
                "Fire Blast",
                "Stealth Rock"
              ],
              "ability": "Rough Skin"
            },
            {
              "pokemon": "Alakazam",
              "item": "Focus Sash",
              "moves": [
                "Psychic",
                "Dazzling Gleam",
                "Shadow Ball",
                "Nasty Plot"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Weavile",
              "item": "Life Orb",
              "moves": [
                "Knock Off",
                "Ice Shard",
                "Pursuit",
                "Triple Axel"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Melmetal",
              "item": "Assault Vest",
              "moves": [
                "Double Iron Bash",
                "Body Press",
                "Ice Punch",
                "Thunder Punch"
              ],
              "ability": "Iron Fist"
            },
            {
              "pokemon": "Zapdos_Galarian",
              "item": "Iapapa Berry",
              "moves": [
                "Brave Bird",
                "Thunderous Kick",
                "Throat Chop",
                "Bulk Up"
              ],
              "ability": "Defiant"
            },
            {
              "pokemon": "Swampert",
              "item": "Swampertite",
              "moves": [
                "Earthquake",
                "Liquidation",
                "Ice Punch",
                "Power Up Punch"
              ],
              "ability": "Torrent"
            }
          ]
        }
      ]
    },
    "Mt. Pyre": {
      "zone_name": "Mt. Pyre",
      "zone_trainers": [
        {
          "trainer": "Young Couple Dez & Luke [Double]",
          "pokemon_list": [
            {
              "pokemon": "Perrserker",
              "item": "Iapapa Berry",
              "moves": [
                "Iron Tail",
                "Bullet Punch",
                "Close Combat",
                "Fake Out"
              ],
              "ability": "Steely Spirit"
            },
            {
              "pokemon": "Meowstic",
              "item": "Light Clay",
              "moves": [
                "Psychic",
                "Shadow Ball",
                "Light Screen",
                "Reflect"
              ],
              "ability": "Prankster"
            },
            {
              "pokemon": "Arcanine",
              "item": "Fire Gem",
              "moves": [
                "Burn Up",
                "Flare Blitz",
                "Close Combat",
                "Extreme Speed"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Smeargle",
              "item": "Leftovers",
              "moves": [
                "Psywave",
                "Dark Void",
                "Protect",
                "Substitute"
              ],
              "ability": "Moody"
            },
            {
              "pokemon": "Raticate_Alolan",
              "item": "Dark Gem",
              "moves": [
                "Double Edge",
                "Sucker Punch",
                "Zen Headbutt",
                "Stomping Tantrum"
              ],
              "ability": "Hustle"
            }
          ]
        },
        {
          "trainer": "Hex Maniac Leah",
          "pokemon_list": [
            {
              "pokemon": "Hariyama",
              "item": "Flame Orb",
              "moves": [
                "Close Combat",
                "Knock Off",
                "Fake Out",
                "Heavy Slam"
              ],
              "ability": "Guts"
            },
            {
              "pokemon": "Umbreon",
              "item": "Leftovers",
              "moves": [
                "Foul Play",
                "Payback",
                "Baton Pass",
                "Curse"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Beheeyem",
              "item": "Power Herb",
              "moves": [
                "Psychic",
                "Meteor Beam",
                "Signal Beam",
                "Recover"
              ],
              "ability": "Analytic"
            },
            {
              "pokemon": "Delphox",
              "item": "White Herb",
              "moves": [
                "Flamethrower",
                "Psychic",
                "Stored Power",
                "Calm Mind"
              ],
              "ability": "Magic Guard"
            }
          ]
        },
        {
          "trainer": "Pok\u00e9maniac Mark",
          "pokemon_list": [
            {
              "pokemon": "Cramorant",
              "item": "Wacan Berry",
              "moves": [
                "Hurricane",
                "Surf",
                "Belch",
                "Roost"
              ],
              "ability": "Gulp Missile"
            },
            {
              "pokemon": "Rhydon",
              "item": "Eviolite",
              "moves": [
                "Head Smash",
                "Earthquake",
                "Counter",
                "Stealth Rock"
              ],
              "ability": "Rock Head"
            },
            {
              "pokemon": "Hitmonlee",
              "item": "Normal Gem",
              "moves": [
                "Close Combat",
                "Double Edge",
                "Knock Off",
                "Fake Out"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Yanmega",
              "item": "Rock Gem",
              "moves": [
                "Bug Buzz",
                "Air Slash",
                "Ancient Power",
                "Hypnosis"
              ],
              "ability": "Speed Boost"
            }
          ]
        },
        {
          "trainer": "Psychic William",
          "pokemon_list": [
            {
              "pokemon": "Runerigus",
              "item": "Custap Berry",
              "moves": [
                "Poltergeist",
                "Body Press",
                "Destiny Bond",
                "Toxic Spikes"
              ],
              "ability": "Wandering Spirit"
            },
            {
              "pokemon": "Tentacruel",
              "item": "Black Sludge",
              "moves": [
                "Scald",
                "Venoshock",
                "Hex",
                "Toxic"
              ],
              "ability": "Liquid Ooze"
            },
            {
              "pokemon": "Exeggutor",
              "item": "Sitrus Berry",
              "moves": [
                "Psychic",
                "Giga Drain",
                "Leech Seed",
                "Substitute"
              ],
              "ability": "Harvest"
            },
            {
              "pokemon": "Glaceon",
              "item": "Leftovers",
              "moves": [
                "Blizzard",
                "Freeze Dry",
                "Mirror Coat",
                "Barrier"
              ],
              "ability": "Snow Cloak"
            }
          ]
        },
        {
          "trainer": "Pokemon Breeder Gabrielle",
          "pokemon_list": [
            {
              "pokemon": "Clefable",
              "item": "Lum Berry",
              "moves": [
                "Moonblast",
                "Fire Blast",
                "Psychic",
                "Stealth Rock"
              ],
              "ability": "Magic Guard"
            },
            {
              "pokemon": "Breloom",
              "item": "Focus Sash",
              "moves": [
                "Force Palm",
                "Mach Punch",
                "Bullet Seed",
                "Swords Dance"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Froslass",
              "item": "Ice Gem",
              "moves": [
                "Blizzard",
                "Shadow Ball",
                "Sing",
                "Will O Wisp"
              ],
              "ability": "Cursed Body"
            },
            {
              "pokemon": "Raichu_Alolan",
              "item": "Electric Gem",
              "moves": [
                "Thunder",
                "Psychic",
                "Grass Knot",
                "Sing"
              ],
              "ability": "Surge Surfer"
            },
            {
              "pokemon": "Grimmsnarl",
              "item": "Leftovers",
              "moves": [
                "Play Rough",
                "Sucker Punch",
                "Body Press",
                "Bulk Up"
              ],
              "ability": "Prankster"
            }
          ]
        },
        {
          "trainer": "Hex Maniac Tasha",
          "pokemon_list": [
            {
              "pokemon": "Omastar",
              "item": "Focus Sash",
              "moves": [
                "Surf",
                "Power Gem",
                "Spikes",
                "Stealth Rock"
              ],
              "ability": "Weak Armor"
            },
            {
              "pokemon": "Muk_Alolan",
              "item": "Black Sludge",
              "moves": [
                "Poison Jab",
                "Payback",
                "Brick Break",
                "Curse"
              ],
              "ability": "Poison Touch"
            },
            {
              "pokemon": "Carnivine",
              "item": "Grass Gem",
              "moves": [
                "Leaf Storm",
                "Power Whip",
                "Knock Off",
                "Infestation"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Spiritomb",
              "item": "Dark Gem",
              "moves": [
                "Dark Pulse",
                "Shadow Ball",
                "Pursuit",
                "Calm Mind"
              ],
              "ability": "Pressure"
            },
            {
              "pokemon": "Dusclops",
              "item": "Eviolite",
              "moves": [
                "Shadow Punch",
                "Power Up Punch",
                "Pain Split",
                "Will O Wisp"
              ],
              "ability": "Pressure"
            }
          ]
        },
        {
          "trainer": "Black Belt Atsushi",
          "pokemon_list": [
            {
              "pokemon": "Krookodile",
              "item": "Lum Berry",
              "moves": [
                "Earthquake",
                "Foul Play",
                "Close Combat",
                "Rock Tomb"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Lycanroc_Dusk",
              "item": "Rock Gem",
              "moves": [
                "Accelerock",
                "Close Combat",
                "Crunch",
                "Stealth Rock"
              ],
              "ability": "Tough Claws"
            },
            {
              "pokemon": "Infernape",
              "item": "Focus Sash",
              "moves": [
                "Flamethrower",
                "Vacuum Wave",
                "Grass Knot",
                "Nasty Plot"
              ],
              "ability": "Blaze"
            },
            {
              "pokemon": "Poliwrath",
              "item": "Leftovers",
              "moves": [
                "Close Combat",
                "Liquidation",
                "Double Team",
                "Hypnosis"
              ],
              "ability": "Water Absorb"
            },
            {
              "pokemon": "Chesnaught",
              "item": "Salac Berry",
              "moves": [
                "Wood Hammer",
                "Drain Punch",
                "Shadow Claw",
                "Swords Dance"
              ],
              "ability": "Overgrow"
            }
          ]
        },
        {
          "trainer": "Hex Maniac Valerie",
          "pokemon_list": [
            {
              "pokemon": "Kangaskhan",
              "item": "Assault Vest",
              "moves": [
                "Double Edge",
                "Earthquake",
                "Sucker Punch",
                "Counter"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Mismagius",
              "item": "Life Orb",
              "moves": [
                "Shadow Ball",
                "Energy Ball",
                "Mystical Fire",
                "Destiny Bond"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Malamar",
              "item": "Leftovers",
              "moves": [
                "Night Slash",
                "Psycho Cut",
                "Superpower",
                "Substitute"
              ],
              "ability": "Contrary"
            },
            {
              "pokemon": "Wobbuffet",
              "item": "Lum Berry",
              "moves": [
                "Counter",
                "Mirror Coat",
                "Destiny Bond",
                "Encore"
              ],
              "ability": "Shadow Tag"
            }
          ]
        },
        {
          "trainer": "Psychic Cedric",
          "pokemon_list": [
            {
              "pokemon": "Electivire",
              "item": "Life Orb",
              "moves": [
                "Wild Charge",
                "Psychic",
                "Ice Punch",
                "Hidden Power Grass"
              ],
              "ability": "Vital Spirit"
            },
            {
              "pokemon": "Obstagoon",
              "item": "Roseli Berry",
              "moves": [
                "Double Edge",
                "Night Slash",
                "Close Combat",
                "Bulk Up"
              ],
              "ability": "Reckless"
            },
            {
              "pokemon": "Polteageist",
              "item": "White Herb",
              "moves": [
                "Shadow Ball",
                "Self Destruct",
                "Stored Power",
                "Shell Smash"
              ],
              "ability": "Cursed Body"
            },
            {
              "pokemon": "Gothitelle",
              "item": "Expert Belt",
              "moves": [
                "Psychic",
                "Energy Ball",
                "Thunderbolt",
                "Shadow Ball"
              ],
              "ability": "Shadow Tag"
            }
          ]
        },
        {
          "trainer": "Psychic Kayla",
          "pokemon_list": [
            {
              "pokemon": "Golduck",
              "item": "Life Orb",
              "moves": [
                "Hydro Pump",
                "Focus Blast",
                "Ice Beam",
                "Psychic"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Bronzong",
              "item": "Colbur Berry",
              "moves": [
                "Gyro Ball",
                "Body Press",
                "Payback",
                "Iron Defense"
              ],
              "ability": "Heatproof"
            },
            {
              "pokemon": "Corsola_Galarian",
              "item": "Eviolite",
              "moves": [
                "Night Shade",
                "Strength Sap",
                "Toxic",
                "Will O Wisp"
              ],
              "ability": "Cursed Body"
            },
            {
              "pokemon": "Mandibuzz",
              "item": "Rocky Helmet",
              "moves": [
                "Foul Play",
                "Knock Off",
                "Infestation",
                "Roost"
              ],
              "ability": "Overcoat"
            }
          ]
        },
        {
          "trainer": "Black Belt Zander",
          "pokemon_list": [
            {
              "pokemon": "Sawk",
              "item": "Lum Berry",
              "moves": [
                "Close Combat",
                "Reversal",
                "Knock Off",
                "Counter"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Throh",
              "item": "Leftovers",
              "moves": [
                "Revenge",
                "Payback",
                "Bulk Up",
                "Recover"
              ],
              "ability": "Guts"
            },
            {
              "pokemon": "Lucario",
              "item": "Normal Gem",
              "moves": [
                "Focus Blast",
                "Flash Cannon",
                "Stone Edge",
                "Extreme Speed"
              ],
              "ability": "Steadfast"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Crawdaunt",
              "item": "Focus Sash",
              "moves": [
                "Crabhammer",
                "Knock Off",
                "Aqua Jet",
                "Close Combat"
              ],
              "ability": "Adaptability"
            },
            {
              "pokemon": "Kingdra",
              "item": "Scope Lens",
              "moves": [
                "Draco Meteor",
                "Liquidation",
                "Octazooka",
                "Focus Energy"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Gumshoos",
              "item": "Quick Claw",
              "moves": [
                "Frustration",
                "Earthquake",
                "Crunch",
                "Endeavor"
              ],
              "ability": "Adaptability"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Dragonite",
              "item": "White Herb",
              "moves": [
                "Draco Meteor",
                "Dual Wingbeat",
                "Earthquake",
                "Roost"
              ],
              "ability": "Multiscale"
            },
            {
              "pokemon": "Avalugg",
              "item": "Lum Berry",
              "moves": [
                "Avalanche",
                "Stone Edge",
                "Body Press",
                "Mirror Coat"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Floatzel",
              "item": "Life Orb",
              "moves": [
                "Hydro Pump",
                "Waterfall",
                "Ice Beam",
                "Low Kick"
              ],
              "ability": "Water Veil"
            },
            {
              "pokemon": "Gyarados",
              "item": "Bright Powder",
              "moves": [
                "Aqua Tail",
                "Crunch",
                "Ice Fang",
                "Dragon Dance"
              ],
              "ability": "Intimidate"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Gengar",
              "item": "Focus Band",
              "moves": [
                "Sludge Wave",
                "Shadow Ball",
                "Explosion",
                "Dazzling Gleam"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Octillery",
              "item": "Water Gem",
              "moves": [
                "Water Spout",
                "Octazooka",
                "Sludge Wave",
                "Energy Ball"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Skuntank",
              "item": "Lum Berry",
              "moves": [
                "Sludge Bomb",
                "Dark Pulse",
                "Explosion",
                "Nasty Plot"
              ],
              "ability": "Aftermath"
            },
            {
              "pokemon": "Relicanth",
              "item": "Bright Powder",
              "moves": [
                "Head Smash",
                "Aqua Tail",
                "Whirlpool",
                "Earthquake"
              ],
              "ability": "Rock Head"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Inteleon",
              "item": "Light Clay",
              "moves": [
                "Snipe Shot",
                "Air Slash",
                "Light Screen",
                "Reflect"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Luxray",
              "item": "Flame Orb",
              "moves": [
                "Zing Zap",
                "Superpower",
                "Crunch",
                "Facade"
              ],
              "ability": "Guts"
            },
            {
              "pokemon": "Cloyster",
              "item": "Kings Rock",
              "moves": [
                "Liquidation",
                "Icicle Spear",
                "Rock Blast",
                "Shell Smash"
              ],
              "ability": "Skill Link"
            },
            {
              "pokemon": "Drapion",
              "item": "Scope Lens",
              "moves": [
                "Cross Poison",
                "Night Slash",
                "Aqua Tail",
                "Swords Dance"
              ],
              "ability": "Sniper"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt [Double Battle With Team Aqua Grunt]",
          "pokemon_list": [
            {
              "pokemon": "Toxicroak",
              "item": "Focus Band",
              "moves": [
                "Sludge Bomb",
                "Aura Sphere",
                "Fake Out",
                "Helping Hand"
              ],
              "ability": "Poison Touch"
            },
            {
              "pokemon": "Raichu",
              "item": "Focus Band",
              "moves": [
                "Thunderbolt",
                "Grass Knot",
                "Fake Out",
                "Helping Hand"
              ],
              "ability": "Lightning Rod"
            },
            {
              "pokemon": "Dracovish",
              "item": "Lum Berry",
              "moves": [
                "Dragon Rush",
                "Fishious Rend",
                "Psychic Fangs",
                "Crunch"
              ],
              "ability": "Strong Jaw"
            },
            {
              "pokemon": "Basculin",
              "item": "Choice Band",
              "moves": [
                "Liquidation"
              ],
              "ability": "Adaptability"
            }
          ]
        },
        {
          "trainer": "Aqua Leader Archie [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Urshifu_Rapid_Strike_Style",
              "item": "Lum Berry",
              "moves": [
                "Close Combat",
                "Surging Strikes",
                "Poison Jab",
                "Detect"
              ],
              "ability": "Unseen Fist"
            },
            {
              "pokemon": "Hydreigon",
              "item": "Dragon Gem",
              "moves": [
                "Dragon Pulse",
                "Dark Pulse",
                "Aura Sphere",
                "Roost"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Rotom_Frost",
              "item": "Leftovers",
              "moves": [
                "Blizzard",
                "Thunderbolt",
                "Light Screen",
                "Reflect"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Dragalge",
              "item": "Black Sludge",
              "moves": [
                "Dragon Pulse",
                "Sludge Bomb",
                "Focus Blast",
                "Protect"
              ],
              "ability": "Adaptability"
            },
            {
              "pokemon": "Stakataka",
              "item": "Assault Vest",
              "moves": [
                "Heavy Slam",
                "Stone Edge",
                "High Horsepower",
                "Body Press"
              ],
              "ability": "Beast Boost"
            },
            {
              "pokemon": "Sharpedo",
              "item": "Sharpedonite",
              "moves": [
                "Waterfall",
                "Crunch",
                "Close Combat",
                "Ice Fang"
              ],
              "ability": "Rough Skin"
            }
          ]
        },
        {
          "trainer": "Pokemon Trainer Chelle",
          "pokemon_list": [
            {
              "pokemon": "Delcatty",
              "item": "Silk Scarf",
              "moves": [
                "Last Resort",
                "Fake Out"
              ],
              "ability": "Normalize"
            },
            {
              "pokemon": "Rhyperior",
              "item": "Iapapa Berry",
              "moves": [
                "Stone Edge",
                "High Horsepower",
                "Avalanche",
                "Protect"
              ],
              "ability": "Solid Rock"
            },
            {
              "pokemon": "Venusaur",
              "item": "Venusaurite",
              "moves": [
                "Sludge Bomb",
                "Giga Drain",
                "Stomping Tantrum",
                "Sleep Powder"
              ],
              "ability": "Overgrow"
            }
          ]
        }
      ]
    },
    "Magma Hideout": {
      "zone_name": "Magma Hideout",
      "zone_trainers": [
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Torkoal",
              "item": "Quick Claw",
              "moves": [
                "Fire Blast",
                "Solar Beam",
                "Explosion",
                "Shell Smash"
              ],
              "ability": "Drought"
            },
            {
              "pokemon": "Tauros",
              "item": "Life Orb",
              "moves": [
                "Body Slam",
                "Iron Tail",
                "Play Rough",
                "Fire Blast"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Charizard",
              "item": "Bright Powder",
              "moves": [
                "Dual Wingbeat",
                "Flame Charge",
                "Earthquake",
                "Belly Drum"
              ],
              "ability": "Blaze"
            },
            {
              "pokemon": "Granbull",
              "item": "Assault Vest",
              "moves": [
                "Play Rough",
                "Earthquake",
                "Fire Punch",
                "Payback"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Arcanine_Hisuian",
              "item": "Focus Band",
              "moves": [
                "Head Smash",
                "Flare Blitz",
                "Reversal",
                "Howl"
              ],
              "ability": "Rock Head"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Aggron",
              "item": "Focus Band",
              "moves": [
                "Stone Edge",
                "Heavy Slam",
                "Counter",
                "Stealth Rock"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Seviper",
              "item": "Quick Claw",
              "moves": [
                "Gunk Shot",
                "Fire Blast",
                "Earthquake",
                "Giga Drain"
              ],
              "ability": "Shed Skin"
            },
            {
              "pokemon": "Electivire",
              "item": "Scope Lens",
              "moves": [
                "Wild Charge",
                "Cross Chop",
                "Ice Punch",
                "Magnet Rise"
              ],
              "ability": "Vital Spirit"
            },
            {
              "pokemon": "Marowak_Alolan",
              "item": "Thick Club",
              "moves": [
                "Flare Blitz",
                "Shadow Bone",
                "Earthquake",
                "Substitute"
              ],
              "ability": "Rock Head"
            },
            {
              "pokemon": "Houndoom",
              "item": "Dark Gem",
              "moves": [
                "Flamethrower",
                "Dark Pulse",
                "Scorching Sands",
                "Will O Wisp"
              ],
              "ability": "Unnerve"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Nidoking",
              "item": "Focus Band",
              "moves": [
                "Earthquake",
                "Poison Jab",
                "Ice Punch",
                "Stealth Rock"
              ],
              "ability": "Poison Point"
            },
            {
              "pokemon": "Camerupt",
              "item": "Quick Claw",
              "moves": [
                "Eruption",
                "Earthquake",
                "Explosion",
                "Stone Edge"
              ],
              "ability": "Own Tempo"
            },
            {
              "pokemon": "Aerodactyl",
              "item": "Kings Rock",
              "moves": [
                "Dual Wingbeat",
                "Rock Blast",
                "Earthquake",
                "Fire Fang"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Slaking",
              "item": "Eject Button",
              "moves": [
                "Double Edge",
                "Retaliate",
                "Earthquake",
                "Crunch"
              ],
              "ability": "Truant"
            },
            {
              "pokemon": "Zoroark",
              "item": "Dark Gem",
              "moves": [
                "Pursuit",
                "Fire Blast",
                "Focus Blast",
                "Sludge Bomb"
              ],
              "ability": "Illusion"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Manectric",
              "item": "Life Orb",
              "moves": [
                "Thunderbolt",
                "Flamethrower",
                "Hidden Power Ice",
                "Double Team"
              ],
              "ability": "Static"
            },
            {
              "pokemon": "Pyroar",
              "item": "Focus Band",
              "moves": [
                "Fire Blast",
                "Hyper Voice",
                "Fire Spin",
                "Yawn"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Fearow",
              "item": "Scope Lens",
              "moves": [
                "Brave Bird",
                "Double Edge",
                "Quick Attack",
                "Drill Run"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Absol",
              "item": "Scope Lens",
              "moves": [
                "Night Slash",
                "Stone Edge",
                "Psycho Cut",
                "Fire Blast"
              ],
              "ability": "Super Luck"
            },
            {
              "pokemon": "Trapinch",
              "item": "Eviolite",
              "moves": [
                "Earthquake",
                "Crunch",
                "Leech Life",
                "Swagger"
              ],
              "ability": "Arena Trap"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Druddigon",
              "item": "Roseli Berry",
              "moves": [
                "Dragon Claw",
                "Gunk Shot",
                "Fire Blast",
                "Glare"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Shiftry",
              "item": "Bright Powder",
              "moves": [
                "Leaf Storm",
                "Leaf Blade",
                "Night Slash",
                "Low Kick"
              ],
              "ability": "Early Bird"
            },
            {
              "pokemon": "Marowak",
              "item": "Thick Club",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Fire Punch",
                "Knock Off"
              ],
              "ability": "Battle Armor"
            },
            {
              "pokemon": "Turtonator",
              "item": "Quick Claw",
              "moves": [
                "Flamethrower",
                "Dragon Pulse",
                "Explosion",
                "Shell Smash"
              ],
              "ability": "Shell Armor"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Coalossal",
              "item": "Quick Claw",
              "moves": [
                "Burn Up",
                "Stone Edge",
                "Explosion",
                "Stealth Rock"
              ],
              "ability": "Flame Body"
            },
            {
              "pokemon": "Flareon",
              "item": "Toxic Orb",
              "moves": [
                "Flare Blitz",
                "Flame Charge",
                "Superpower",
                "Facade"
              ],
              "ability": "Guts"
            },
            {
              "pokemon": "Solrock",
              "item": "Bright Powder",
              "moves": [
                "Stone Edge",
                "Zen Headbutt",
                "Explosion",
                "Will O Wisp"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Dracozolt",
              "item": "Fire Gem",
              "moves": [
                "Bolt Beak",
                "Dragon Claw",
                "Iron Tail",
                "Fire Blast"
              ],
              "ability": "Hustle"
            },
            {
              "pokemon": "Talonflame",
              "item": "Focus Band",
              "moves": [
                "Brave Bird",
                "Roost",
                "Swords Dance",
                "Will O Wisp"
              ],
              "ability": "Flame Body"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt [Double Battle With Team Magma Grunt]",
          "pokemon_list": [
            {
              "pokemon": "Ninetales",
              "item": "Fire Gem",
              "moves": [
                "Heat Wave",
                "Flamethrower",
                "Scorching Sands",
                "Imprison"
              ],
              "ability": "Drought"
            },
            {
              "pokemon": "Miltank",
              "item": "Leftovers",
              "moves": [
                "Return",
                "Body Press",
                "Curse",
                "Milk Drink"
              ],
              "ability": "Thick Fat"
            },
            {
              "pokemon": "Donphan",
              "item": "Custap Berry",
              "moves": [
                "High Horsepower",
                "Stone Edge",
                "Flail",
                "Natural Gift"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Exeggutor",
              "item": "Tanga Berry",
              "moves": [
                "Solar Beam",
                "Psychic",
                "Ancient Power",
                "Hidden Power Fire"
              ],
              "ability": "Chlorophyll"
            },
            {
              "pokemon": "Jumpluff",
              "item": "Focus Sash",
              "moves": [
                "Solar Beam",
                "Air Slash",
                "Growth",
                "Spore"
              ],
              "ability": "Leaf Guard"
            },
            {
              "pokemon": "Arcanine",
              "item": "Iapapa Berry",
              "moves": [
                "Flare Blitz",
                "Close Combat",
                "Wild Charge",
                "Scary Face"
              ],
              "ability": "Intimidate"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Togekiss",
              "item": "Scope Lens",
              "moves": [
                "Dazzling Gleam",
                "Air Slash",
                "Flamethrower",
                "Aura Sphere"
              ],
              "ability": "Super Luck"
            },
            {
              "pokemon": "Seismitoad",
              "item": "Bright Powder",
              "moves": [
                "Earthquake",
                "Muddy Water",
                "Sludge Wave",
                "Stealth Rock"
              ],
              "ability": "Water Absorb"
            },
            {
              "pokemon": "Slowbro",
              "item": "Quick Claw",
              "moves": [
                "Psychic",
                "Flamethrower",
                "Grass Knot",
                "Calm Mind"
              ],
              "ability": "Regenerator"
            },
            {
              "pokemon": "Electrode",
              "item": "Bright Powder",
              "moves": [
                "Thunderbolt",
                "Explosion",
                "Foul Play",
                "Swagger"
              ],
              "ability": "Static"
            },
            {
              "pokemon": "Dhelmise",
              "item": "Assault Vest",
              "moves": [
                "Phantom Force",
                "Giga Drain",
                "Anchor Shot",
                "Whirlpool"
              ],
              "ability": "Steelworker"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Silvally_Fire",
              "item": "Fire Memory",
              "moves": [
                "Multi Attack",
                "Explosion",
                "Thunderbolt",
                "Grass Pledge"
              ],
              "ability": "RKS System"
            },
            {
              "pokemon": "Zoroark",
              "item": "Focus Sash",
              "moves": [
                "Night Daze",
                "Fire Blast",
                "Grass Knot",
                "Nasty Plot"
              ],
              "ability": "Illusion"
            },
            {
              "pokemon": "Flygon",
              "item": "Bright Powder",
              "moves": [
                "Earthquake",
                "Scale Shot",
                "Fire Punch",
                "Dragon Dance"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Skuntank",
              "item": "Bright Powder",
              "moves": [
                "Sludge Bomb",
                "Dark Pulse",
                "Explosion",
                "Fire Blast"
              ],
              "ability": "Aftermath"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Weezing_Galarian",
              "item": "Payapa Berry",
              "moves": [
                "Misty Explosion",
                "Sludge Wave",
                "Strange Steam",
                "Fire Blast"
              ],
              "ability": "Misty Surge"
            },
            {
              "pokemon": "Muk_Alolan",
              "item": "Shuca Berry",
              "moves": [
                "Gunk Shot",
                "Knock Off",
                "Pursuit",
                "Explosion"
              ],
              "ability": "Poison Touch"
            },
            {
              "pokemon": "Cinderace",
              "item": "Misty Seed",
              "moves": [
                "Pyro Ball",
                "Acrobatics",
                "Low Kick",
                "Bulk Up"
              ],
              "ability": "Libero"
            },
            {
              "pokemon": "Drampa",
              "item": "Quick Claw",
              "moves": [
                "Hyper Voice",
                "Flamethrower",
                "Psychic",
                "Calm Mind"
              ],
              "ability": "Berserk"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Runerigus",
              "item": "Bright Powder",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Stealth Rock",
                "Toxic Spikes"
              ],
              "ability": "Wandering Spirit"
            },
            {
              "pokemon": "Slowking_Galarian",
              "item": "Black Sludge",
              "moves": [
                "Venoshock",
                "Fire Blast",
                "Grass Knot",
                "Protect"
              ],
              "ability": "Regenerator"
            },
            {
              "pokemon": "Chandelure",
              "item": "Leftovers",
              "moves": [
                "Flamethrower",
                "Hex",
                "Substitute",
                "Will O Wisp"
              ],
              "ability": "Shadow Tag"
            },
            {
              "pokemon": "Cacturne",
              "item": "Quick Claw",
              "moves": [
                "Dark Pulse",
                "Seed Bomb",
                "Drain Punch",
                "Growth"
              ],
              "ability": "Water Absorb"
            },
            {
              "pokemon": "Ninjask",
              "item": "Focus Band",
              "moves": [
                "Dual Wingbeat",
                "Leech Life",
                "Protect",
                "Swords Dance"
              ],
              "ability": "Speed Boost"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Kangaskhan",
              "item": "Silk Scarf",
              "moves": [
                "Return",
                "Fake Out",
                "Low Kick",
                "Sing"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Centiskorch",
              "item": "Leftovers",
              "moves": [
                "Fire Lash",
                "Leech Life",
                "Coil",
                "Protect"
              ],
              "ability": "Flame Body"
            },
            {
              "pokemon": "Crobat",
              "item": "Choice Specs",
              "moves": [
                "Hurricane",
                "Sludge Bomb",
                "Heat Wave",
                "Hidden Power Grass"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Toxapex",
              "item": "Black Sludge",
              "moves": [
                "Liquidation",
                "Venoshock",
                "Toxic",
                "Recover"
              ],
              "ability": "Merciless"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Grimmsnarl",
              "item": "Kebia Berry",
              "moves": [
                "Play Rough",
                "Darkest Lariat",
                "Fake Out",
                "Scary Face"
              ],
              "ability": "Prankster"
            },
            {
              "pokemon": "Turtonator",
              "item": "Eject Pack",
              "moves": [
                "Draco Meteor",
                "Overheat",
                "Scorching Sands",
                "Protect"
              ],
              "ability": "Shell Armor"
            },
            {
              "pokemon": "Lickilicky",
              "item": "Quick Claw",
              "moves": [
                "Explosion"
              ],
              "ability": "Own Tempo"
            },
            {
              "pokemon": "Incineroar",
              "item": "Assault Vest",
              "moves": [
                "Blaze Kick",
                "Knock Off",
                "Earthquake",
                "Fake Out"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Jumpluff",
              "item": "Leftovers",
              "moves": [
                "Air Slash",
                "Leech Seed",
                "Spore",
                "Substitute"
              ],
              "ability": "Infiltrator"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Galvantula",
              "item": "Focus Sash",
              "moves": [
                "Thunder",
                "Bug Buzz",
                "Energy Ball",
                "Sticky Web"
              ],
              "ability": "Compound Eyes"
            },
            {
              "pokemon": "Magmortar",
              "item": "Choice Specs",
              "moves": [
                "Flamethrower",
                "Psychic",
                "Thunderbolt",
                "Scorching Sands"
              ],
              "ability": "Flame Body"
            },
            {
              "pokemon": "Ursaring",
              "item": "Toxic Orb",
              "moves": [
                "Facade",
                "Earthquake",
                "Crunch",
                "Protect"
              ],
              "ability": "Quick Feet"
            },
            {
              "pokemon": "Arbok",
              "item": "Bright Powder",
              "moves": [
                "Gunk Shot",
                "Aqua Tail",
                "Coil",
                "Glare"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Carnivine",
              "item": "Scope Lens",
              "moves": [
                "Leaf Storm",
                "Leaf Blade",
                "Knock Off",
                "Sleep Powder"
              ],
              "ability": "Levitate"
            }
          ]
        },
        {
          "trainer": "Magma Admin Tabitha",
          "pokemon_list": [
            {
              "pokemon": "Blacephalon",
              "item": "Focus Sash",
              "moves": [
                "Flamethrower",
                "Shadow Ball",
                "Explosion",
                "Psyshock"
              ],
              "ability": "Beast Boost"
            },
            {
              "pokemon": "Hitmonlee",
              "item": "Normal Gem",
              "moves": [
                "Low Kick",
                "Earthquake",
                "Stone Edge",
                "Fake Out"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Yanmega",
              "item": "Flying Gem",
              "moves": [
                "Air Slash",
                "Giga Drain",
                "Hidden Power Fire",
                "Detect"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Snorlax",
              "item": "Leftovers",
              "moves": [
                "Body Slam",
                "Heat Crash",
                "Curse",
                "Rest"
              ],
              "ability": "Thick Fat"
            },
            {
              "pokemon": "Victini",
              "item": "Eject Pack",
              "moves": [
                "V Create",
                "Psychic",
                "Bolt Strike",
                "Energy Ball"
              ],
              "ability": "Victory Star"
            },
            {
              "pokemon": "Absol",
              "item": "Absolite",
              "moves": [
                "Night Slash",
                "Sucker Punch",
                "Close Combat",
                "Swords Dance"
              ],
              "ability": "Pressure"
            }
          ]
        },
        {
          "trainer": "Magma Leader Maxie [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Groudon",
              "item": "Iapapa Berry",
              "moves": [
                "Precipice Blades",
                "Stone Edge",
                "Heat Crash",
                "Rock Polish"
              ],
              "ability": "Drought"
            },
            {
              "pokemon": "Tangrowth",
              "item": "Life Orb",
              "moves": [
                "Solar Beam",
                "Hidden Power Ice",
                "Weather Ball",
                "Sleep Powder"
              ],
              "ability": "Chlorophyll"
            },
            {
              "pokemon": "Garchomp",
              "item": "Rocky Helmet",
              "moves": [
                "Earthquake",
                "Scale Shot",
                "Stone Edge",
                "Swords Dance"
              ],
              "ability": "Rough Skin"
            },
            {
              "pokemon": "Naganadel",
              "item": "Choice Scarf",
              "moves": [
                "Sludge Wave",
                "Dragon Pulse",
                "Flamethrower",
                "U Turn"
              ],
              "ability": "Beast Boost"
            },
            {
              "pokemon": "Mew",
              "item": "Lum Berry",
              "moves": [
                "Psychic Fangs",
                "Flare Blitz",
                "Dragon Dance",
                "Soft Boiled"
              ],
              "ability": "Synchronize"
            },
            {
              "pokemon": "Houndoom",
              "item": "Houndoominite",
              "moves": [
                "Flamethrower",
                "Dark Pulse",
                "Solar Beam",
                "Destiny Bond"
              ],
              "ability": "Flash Fire"
            }
          ]
        }
      ]
    },
    "Aqua Hideout": {
      "zone_name": "Aqua Hideout",
      "zone_trainers": [
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Persian_Alolan",
              "item": "Bright Powder",
              "moves": [
                "Night Slash",
                "Gunk Shot",
                "Play Rough",
                "Hypnosis"
              ],
              "ability": "Fur Coat"
            },
            {
              "pokemon": "Pangoro",
              "item": "Quick Claw",
              "moves": [
                "Close Combat",
                "Crunch",
                "Gunk Shot",
                "Stone Edge"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Bouffalant",
              "item": "Water Gem",
              "moves": [
                "Head Charge",
                "Surf",
                "Lash Out",
                "Work Up"
              ],
              "ability": "Sap Sipper"
            },
            {
              "pokemon": "Grapploct",
              "item": "Assault Vest",
              "moves": [
                "Revenge",
                "Ice Punch",
                "Payback",
                "Whirlpool"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Floatzel",
              "item": "Focus Band",
              "moves": [
                "Liquidation",
                "Crunch",
                "Ice Punch",
                "Power Up Punch"
              ],
              "ability": "Water Veil"
            },
            {
              "pokemon": "Kingler",
              "item": "Life Orb",
              "moves": [
                "Liquidation",
                "X Scissor",
                "Rock Slide",
                "Agility"
              ],
              "ability": "Sheer Force"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Cryogonal",
              "item": "Focus Band",
              "moves": [
                "Ice Beam",
                "Freeze Dry",
                "Reflect",
                "Light Screen"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Hariyama",
              "item": "Life Orb",
              "moves": [
                "Close Combat",
                "Poison Jab",
                "Rock Slide",
                "Thunder Punch"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Serperior",
              "item": "Bright Powder",
              "moves": [
                "Leaf Storm",
                "Dragon Pulse",
                "Glare",
                "Leech Seed"
              ],
              "ability": "Contrary"
            },
            {
              "pokemon": "Alomomola",
              "item": "Rocky Helmet",
              "moves": [
                "Scald",
                "Whirlpool",
                "Toxic",
                "Wish"
              ],
              "ability": "Regenerator"
            },
            {
              "pokemon": "Eelektross",
              "item": "Leftovers",
              "moves": [
                "Wild Charge",
                "Aqua Tail",
                "Knock Off",
                "Coil"
              ],
              "ability": "Levitate"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Omastar",
              "item": "Focus Sash",
              "moves": [
                "Power Gem",
                "Earth Power",
                "Spikes",
                "Stealth Rock"
              ],
              "ability": "Weak Armor"
            },
            {
              "pokemon": "Escavalier",
              "item": "Quick Claw",
              "moves": [
                "Megahorn",
                "Iron Head",
                "Drill Run",
                "Swords Dance"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Lanturn",
              "item": "Bright Powder",
              "moves": [
                "Hydro Pump",
                "Thunder",
                "Confuse Ray",
                "Thunder Wave"
              ],
              "ability": "Volt Absorb"
            },
            {
              "pokemon": "Crobat",
              "item": "Flying Gem",
              "moves": [
                "Acrobatics",
                "Heat Wave",
                "Giga Drain",
                "Super Fang"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Sharpedo",
              "item": "Focus Band",
              "moves": [
                "Waterfall",
                "Dark Pulse",
                "Ice Beam",
                "Earthquake"
              ],
              "ability": "Speed Boost"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Ninetales_Alolan",
              "item": "Light Clay",
              "moves": [
                "Moonblast",
                "Freeze Dry",
                "Aurora Veil",
                "Nasty Plot"
              ],
              "ability": "Snow Warning"
            },
            {
              "pokemon": "Sandslash_Alolan",
              "item": "Scope Lens",
              "moves": [
                "Iron Head",
                "Triple Axel",
                "Earthquake",
                "Swords Dance"
              ],
              "ability": "Slush Rush"
            },
            {
              "pokemon": "Eiscue",
              "item": "Scope Lens",
              "moves": [
                "Icicle Spear",
                "Head Smash",
                "Aqua Jet",
                "Belly Drum"
              ],
              "ability": "Ice Face"
            },
            {
              "pokemon": "Walrein",
              "item": "Bright Powder",
              "moves": [
                "Blizzard",
                "Scald",
                "Substitute",
                "Toxic"
              ],
              "ability": "Ice Body"
            },
            {
              "pokemon": "Clefable",
              "item": "Weakness Policy",
              "moves": [
                "Moonblast",
                "Stored Power",
                "Cosmic Power",
                "Soft Boiled"
              ],
              "ability": "Magic Guard"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Gyarados",
              "item": "Bright Powder",
              "moves": [
                "Waterfall",
                "Earthquake",
                "Stone Edge",
                "Dragon Dance"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Gogoat",
              "item": "Assault Vest",
              "moves": [
                "Horn Leech",
                "Earthquake",
                "Iron Tail",
                "Rock Slide"
              ],
              "ability": "Sap Sipper"
            },
            {
              "pokemon": "Raichu_Alolan",
              "item": "Focus Band",
              "moves": [
                "Psychic",
                "Discharge",
                "Grass Knot",
                "Nasty Plot"
              ],
              "ability": "Surge Surfer"
            },
            {
              "pokemon": "Cursola",
              "item": "Power Herb",
              "moves": [
                "Shadow Ball",
                "Meteor Beam",
                "Hydro Pump",
                "Trick Room"
              ],
              "ability": "Weak Armor"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Politoed",
              "item": "Water Gem",
              "moves": [
                "Ice Beam",
                "Earth Power",
                "Weather Ball",
                "Hypnosis"
              ],
              "ability": "Drizzle"
            },
            {
              "pokemon": "Heliolisk",
              "item": "Bright Powder",
              "moves": [
                "Hyper Voice",
                "Thunderbolt",
                "Weather Ball",
                "Glare"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Noivern",
              "item": "Bright Powder",
              "moves": [
                "Hurricane",
                "Dragon Pulse",
                "Double Team",
                "Roost"
              ],
              "ability": "Infiltrator"
            },
            {
              "pokemon": "Ferrothorn",
              "item": "Leftovers",
              "moves": [
                "Gyro Ball",
                "Explosion",
                "Body Press",
                "Curse"
              ],
              "ability": "Iron Barbs"
            },
            {
              "pokemon": "Poliwrath",
              "item": "Lum Berry",
              "moves": [
                "Liquidation",
                "Drain Punch",
                "Darkest Lariat",
                "Belly Drum"
              ],
              "ability": "Swift Swim"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Carracosta",
              "item": "Quick Claw",
              "moves": [
                "Hydro Pump",
                "Stone Edge",
                "Ice Beam",
                "Stealth Rock"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Octillery",
              "item": "Quick Claw",
              "moves": [
                "Water Spout",
                "Octazooka",
                "Energy Ball",
                "Ice Beam"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Starmie",
              "item": "Life Orb",
              "moves": [
                "Hydro Pump",
                "Psychic",
                "Ice Beam",
                "Thunderbolt"
              ],
              "ability": "Analytic"
            },
            {
              "pokemon": "Gliscor",
              "item": "Toxic Orb",
              "moves": [
                "Earthquake",
                "Dual Wingbeat",
                "Facade",
                "Swords Dance"
              ],
              "ability": "Poison Heal"
            },
            {
              "pokemon": "Goodra",
              "item": "Leftovers",
              "moves": [
                "Dragon Tail",
                "Aqua Tail",
                "Body Press",
                "Curse"
              ],
              "ability": "Gooey"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Rillaboom",
              "item": "Focus Band",
              "moves": [
                "Wood Hammer",
                "Grassy Glide",
                "High Horsepower",
                "Endeavor"
              ],
              "ability": "Grassy Surge"
            },
            {
              "pokemon": "Wobbuffet",
              "item": "Leftovers",
              "moves": [
                "Counter",
                "Mirror Coat",
                "Destiny Bond",
                "Encore"
              ],
              "ability": "Shadow Tag"
            },
            {
              "pokemon": "Tentacruel",
              "item": "Black Sludge",
              "moves": [
                "Sludge Bomb",
                "Scald",
                "Giga Drain",
                "Toxic Spikes"
              ],
              "ability": "Liquid Ooze"
            },
            {
              "pokemon": "Ludicolo",
              "item": "Grass Gem",
              "moves": [
                "Waterfall",
                "Grassy Glide",
                "Zen Headbutt",
                "Swords Dance"
              ],
              "ability": "Own Tempo"
            },
            {
              "pokemon": "Ampharos",
              "item": "Quick Claw",
              "moves": [
                "Thunderbolt",
                "Focus Blast",
                "Hidden Power Ice",
                "Counter"
              ],
              "ability": "Static"
            },
            {
              "pokemon": "Obstagoon",
              "item": "Flame Orb",
              "moves": [
                "Facade",
                "Knock Off",
                "Close Combat",
                "Obstruct"
              ],
              "ability": "Guts"
            }
          ]
        },
        {
          "trainer": "Aqua Admin Matt [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Mamoswine",
              "item": "Focus Sash",
              "moves": [
                "Earthquake",
                "Icicle Crash",
                "Ice Shard",
                "Stealth Rock"
              ],
              "ability": "Oblivious"
            },
            {
              "pokemon": "Dracovish",
              "item": "Choice Scarf",
              "moves": [
                "Fishious Rend"
              ],
              "ability": "Strong Jaw"
            },
            {
              "pokemon": "Dragapult",
              "item": "Dragon Gem",
              "moves": [
                "Shadow Ball",
                "Dragon Darts",
                "Fire Blast",
                "Hydro Pump"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Kartana",
              "item": "Life Orb",
              "moves": [
                "Leaf Blade",
                "Smart Strike",
                "Sacred Sword",
                "Night Slash"
              ],
              "ability": "Beast Boost"
            },
            {
              "pokemon": "Raikou",
              "item": "Water Gem",
              "moves": [
                "Thunderbolt",
                "Scald",
                "Signal Beam",
                "Calm Mind"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Gyarados",
              "item": "Gyaradosite",
              "moves": [
                "Crunch",
                "Waterfall",
                "Earthquake",
                "Dragon Dance"
              ],
              "ability": "Intimidate"
            }
          ]
        }
      ]
    },
    "Route 124 (North)": {
      "zone_name": "Route 124 (North)",
      "zone_trainers": [
        {
          "trainer": "Swimmer\u2640 Grace [Double Battle With Swimmer\u2642 Declan]",
          "pokemon_list": [
            {
              "pokemon": "Greninja",
              "item": "Life Orb",
              "moves": [
                "Dark Pulse",
                "Ice Beam",
                "Grass Knot",
                "Mat Block"
              ],
              "ability": "Protean"
            },
            {
              "pokemon": "Dhelmise",
              "item": "Steel Gem",
              "moves": [
                "Power Whip",
                "Grassy Glide",
                "Shadow Claw",
                "Steel Roller"
              ],
              "ability": "Steelworker"
            },
            {
              "pokemon": "Clawitzer",
              "item": "Focus Sash",
              "moves": [
                "Muddy Water",
                "Water Pulse",
                "Dragon Pulse",
                "Terrain Pulse"
              ],
              "ability": "Mega Launcher"
            },
            {
              "pokemon": "Pincurchin",
              "item": "Electric Gem",
              "moves": [
                "Rising Voltage",
                "Self Destruct",
                "Muddy Water",
                "Thunder Wave"
              ],
              "ability": "Electric Surge"
            },
            {
              "pokemon": "Rillaboom",
              "item": "Assault Vest",
              "moves": [
                "Grassy Glide",
                "High Horsepower",
                "Darkest Lariat",
                "Fake Out"
              ],
              "ability": "Grassy Surge"
            }
          ]
        },
        {
          "trainer": "Swimmer\u2642 Cranberry",
          "pokemon_list": [
            {
              "pokemon": "Slurpuff",
              "item": "Focus Sash",
              "moves": [
                "Misty Explosion",
                "Flamethrower",
                "Thunderbolt",
                "Sticky Web"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Nidoqueen",
              "item": "Life Orb",
              "moves": [
                "Sludge Wave",
                "Earth Power",
                "Blizzard",
                "Fire Blast"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Azumarill",
              "item": "Sitrus Berry",
              "moves": [
                "Play Rough",
                "Aqua Jet",
                "Ice Punch",
                "Belly Drum"
              ],
              "ability": "Huge Power"
            },
            {
              "pokemon": "Snorlax",
              "item": "Custap Berry",
              "moves": [
                "Body Slam",
                "Hammer Arm",
                "Darkest Lariat",
                "Belly Drum"
              ],
              "ability": "Gluttony"
            },
            {
              "pokemon": "Wailord",
              "item": "Water Gem",
              "moves": [
                "Water Spout",
                "Hydro Pump",
                "Self Destruct"
              ],
              "ability": "Oblivious"
            }
          ]
        },
        {
          "trainer": "Triathlete Aubrey",
          "pokemon_list": [
            {
              "pokemon": "Indeedee",
              "item": "Focus Sash",
              "moves": [
                "Expanding Force",
                "Dazzling Gleam",
                "Mystical Fire",
                "Calm Mind"
              ],
              "ability": "Psychic Surge"
            },
            {
              "pokemon": "Primarina",
              "item": "Throat Spray",
              "moves": [
                "Hyper Voice",
                "Draining Kiss",
                "Psychic",
                "Sing"
              ],
              "ability": "Liquid Voice"
            },
            {
              "pokemon": "Flapple",
              "item": "Petaya Berry",
              "moves": [
                "Dragon Claw",
                "Grav Apple",
                "Hidden Power Fire",
                "Dragon Dance"
              ],
              "ability": "Gluttony"
            },
            {
              "pokemon": "Alakazam",
              "item": "Life Orb",
              "moves": [
                "Expanding Force",
                "Dazzling Gleam",
                "Recover",
                "Substitute"
              ],
              "ability": "Magic Guard"
            },
            {
              "pokemon": "Weavile",
              "item": "Dark Gem",
              "moves": [
                "Pursuit",
                "Triple Axel",
                "Psycho Cut",
                "Low Kick"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Slowbro",
              "item": "Psychic Seed",
              "moves": [
                "Liquidation",
                "Body Press",
                "Curse",
                "Slack Off"
              ],
              "ability": "Oblivious"
            }
          ]
        }
      ]
    },
    "Route 124 (North) (Optionals)": {
      "zone_name": "Route 124 (North) (Optionals)",
      "zone_trainers": []
    },
    "Route 125 (Optionals)": {
      "zone_name": "Route 125 (Optionals)",
      "zone_trainers": [
        {
          "trainer": "Bird Keeper Presley [Double Battle With Expert Auron]",
          "pokemon_list": [
            {
              "pokemon": "Dragonite",
              "item": "Lum Berry",
              "moves": [
                "Dragon Rush",
                "Dual Wingbeat",
                "Low Kick",
                "Tailwind"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Kleavor",
              "item": "Life Orb",
              "moves": [
                "Rock Slide",
                "Skitter Smack",
                "Close Combat",
                "Tailwind"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Toucannon",
              "item": "Sitrus Berry",
              "moves": [
                "Brave Bird",
                "Bullet Seed",
                "Rock Blast",
                "Tailwind"
              ],
              "ability": "Skill Link"
            },
            {
              "pokemon": "Salazzle",
              "item": "Focus Sash",
              "moves": [
                "Fire Blast",
                "Sludge Bomb",
                "Fake Out",
                "Encore"
              ],
              "ability": "Corrosion"
            },
            {
              "pokemon": "Rillaboom",
              "item": "Grass Gem",
              "moves": [
                "Wood Hammer",
                "Grassy Glide",
                "Acrobatics",
                "Swords Dance"
              ],
              "ability": "Grassy Surge"
            },
            {
              "pokemon": "Manectric",
              "item": "Manectite",
              "moves": [
                "Thunderbolt",
                "Flamethrower",
                "Hidden Power Grass",
                "Protect"
              ],
              "ability": "Static"
            }
          ]
        },
        {
          "trainer": "Sr. And Jr. Kim & Iris [Double]",
          "pokemon_list": [
            {
              "pokemon": "Ninetales_Alolan",
              "item": "Light Clay",
              "moves": [
                "Blizzard",
                "Moonblast",
                "Icy Wind",
                "Aurora Veil"
              ],
              "ability": "Snow Warning"
            },
            {
              "pokemon": "Medicham",
              "item": "Rock Gem",
              "moves": [
                "Close Combat",
                "Zen Headbutt",
                "Rock Slide",
                "Fake Out"
              ],
              "ability": "Pure Power"
            },
            {
              "pokemon": "Frosmoth",
              "item": "Focus Sash",
              "moves": [
                "Blizzard",
                "Icy Wind",
                "Infestation",
                "Aurora Veil"
              ],
              "ability": "Shield Dust"
            },
            {
              "pokemon": "Jynx",
              "item": "Occa Berry",
              "moves": [
                "Blizzard",
                "Psychic",
                "Aura Sphere",
                "Fake Out"
              ],
              "ability": "Oblivious"
            },
            {
              "pokemon": "Sandslash_Alolan",
              "item": "Life Orb",
              "moves": [
                "Triple Axel",
                "High Horsepower",
                "Knock Off",
                "Swords Dance"
              ],
              "ability": "Slush Rush"
            },
            {
              "pokemon": "Magmortar",
              "item": "Assault Vest",
              "moves": [
                "Heat Wave",
                "Flamethrower",
                "Thunderbolt",
                "Weather Ball"
              ],
              "ability": "Flame Body"
            }
          ]
        }
      ]
    },
    "Mossdeep Gym": {
      "zone_name": "Mossdeep Gym",
      "zone_trainers": [
        {
          "trainer": "Psychic Preston [Double Battle With Psychic Maura]",
          "pokemon_list": [
            {
              "pokemon": "Noivern",
              "item": "Life Orb",
              "moves": [
                "Hurricane",
                "Focus Blast",
                "Super Fang",
                "Tailwind"
              ],
              "ability": "Telepathy"
            },
            {
              "pokemon": "Gardevoir",
              "item": "Lum Berry",
              "moves": [
                "Moonblast",
                "Icy Wind",
                "Shadow Sneak",
                "Helping Hand"
              ],
              "ability": "Telepathy"
            },
            {
              "pokemon": "Mismagius",
              "item": "Focus Sash",
              "moves": [
                "Shadow Ball",
                "Shadow Sneak",
                "Mystical Fire",
                "Psych Up"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Sylveon",
              "item": "Fairy Gem",
              "moves": [
                "Misty Explosion",
                "Hyper Voice",
                "Protect"
              ],
              "ability": "Pixilate"
            },
            {
              "pokemon": "Metagross",
              "item": "Weakness Policy",
              "moves": [
                "Meteor Mash",
                "Zen Headbutt",
                "Bullet Punch",
                "Explosion"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Drifblim",
              "item": "Weakness Policy",
              "moves": [
                "Shadow Ball",
                "Acrobatics",
                "Explosion",
                "Silver Wind"
              ],
              "ability": "Unburden"
            }
          ]
        },
        {
          "trainer": "Psychic Blake [Double Battle With Psychic Samantha]",
          "pokemon_list": [
            {
              "pokemon": "Porygon2",
              "item": "Eviolite",
              "moves": [
                "Tri Attack",
                "Thunder",
                "Psychic",
                "Trick Room"
              ],
              "ability": "Download"
            },
            {
              "pokemon": "Jellicent",
              "item": "Water Gem",
              "moves": [
                "Hydro Pump",
                "Shadow Ball",
                "Strength Sap",
                "Trick Room"
              ],
              "ability": "Water Absorb"
            },
            {
              "pokemon": "Hatterene",
              "item": "Fairy Gem",
              "moves": [
                "Psychic",
                "Dazzling Gleam",
                "Protect",
                "Trick Room"
              ],
              "ability": "Magic Bounce"
            },
            {
              "pokemon": "Perrserker",
              "item": "Occa Berry",
              "moves": [
                "Iron Tail",
                "Close Combat",
                "Fake Out",
                "Metal Sound"
              ],
              "ability": "Steely Spirit"
            },
            {
              "pokemon": "Bronzong",
              "item": "Leftovers",
              "moves": [
                "Heavy Slam",
                "Body Press",
                "Metal Sound",
                "Trick Room"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Rampardos",
              "item": "Choice Band",
              "moves": [
                "Head Smash",
                "Rock Slide"
              ],
              "ability": "Sheer Force"
            }
          ]
        },
        {
          "trainer": "Psychic Virgil [Double Battle With Gentleman Nate]",
          "pokemon_list": [
            {
              "pokemon": "Krookodile",
              "item": "Salac Berry",
              "moves": [
                "High Horsepower",
                "Knock Off",
                "Close Combat",
                "Fling"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Mienshao",
              "item": "Salac Berry",
              "moves": [
                "Close Combat",
                "Stone Edge",
                "Fling",
                "Psych Up"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Delphox",
              "item": "Focus Sash",
              "moves": [
                "Fire Blast",
                "Psychic",
                "Hypnosis",
                "Psych Up"
              ],
              "ability": "Magic Guard"
            },
            {
              "pokemon": "Gothitelle",
              "item": "Weakness Policy",
              "moves": [
                "Psychic",
                "Stored Power",
                "Signal Beam",
                "Fake Out"
              ],
              "ability": "Shadow Tag"
            },
            {
              "pokemon": "Flapple",
              "item": "Liechi Berry",
              "moves": [
                "Dragon Claw",
                "Grav Apple",
                "Sucker Punch",
                "Substitute"
              ],
              "ability": "Ripen"
            },
            {
              "pokemon": "Venomoth",
              "item": "Sitrus Berry",
              "moves": [
                "Bug Buzz",
                "Sludge Bomb",
                "Quiver Dance",
                "Sleep Powder"
              ],
              "ability": "Tinted Lens"
            }
          ]
        },
        {
          "trainer": "Psychic Hannah [Double Battle With Battle Girl Sylvia]",
          "pokemon_list": [
            {
              "pokemon": "Meowstic_Female",
              "item": "Focus Sash",
              "moves": [
                "Psychic",
                "Thunderbolt",
                "Fake Out",
                "Imprison"
              ],
              "ability": "Competitive"
            },
            {
              "pokemon": "Milotic",
              "item": "Leftovers",
              "moves": [
                "Muddy Water",
                "Coil",
                "Hypnosis",
                "Recover"
              ],
              "ability": "Competitive"
            },
            {
              "pokemon": "Purugly",
              "item": "Silk Scarf",
              "moves": [
                "Return",
                "Fake Out",
                "Throat Chop",
                "Leer"
              ],
              "ability": "Defiant"
            },
            {
              "pokemon": "Conkeldurr",
              "item": "Assault Vest",
              "moves": [
                "Hammer Arm",
                "Mach Punch",
                "Poison Jab",
                "Ice Punch"
              ],
              "ability": "Iron Fist"
            },
            {
              "pokemon": "Gyarados",
              "item": "Power Herb",
              "moves": [
                "Bounce",
                "Waterfall",
                "Dragon Dance",
                "Protect"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Rapidash_Galarian",
              "item": "Focus Sash",
              "moves": [
                "Play Rough",
                "Zen Headbutt",
                "High Horsepower",
                "Swords Dance"
              ],
              "ability": "Pastel Veil"
            }
          ]
        },
        {
          "trainer": "Hex Maniac Kathleen [Double Battle With Psychic Nicholas]",
          "pokemon_list": [
            {
              "pokemon": "Crobat",
              "item": "Flying Gem",
              "moves": [
                "Acrobatics",
                "Super Fang",
                "Screech",
                "Tailwind"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Togekiss",
              "item": "Iapapa Berry",
              "moves": [
                "Air Slash",
                "Heat Wave",
                "Follow Me",
                "Tailwind"
              ],
              "ability": "Serene Grace"
            },
            {
              "pokemon": "Jynx",
              "item": "Focus Sash",
              "moves": [
                "Blizzard",
                "Psychic",
                "Fake Out",
                "Lovely Kiss"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Excadrill",
              "item": "Ground Gem",
              "moves": [
                "Earthquake",
                "Iron Head",
                "Rock Slide",
                "Substitute"
              ],
              "ability": "Mold Breaker"
            },
            {
              "pokemon": "Medicham",
              "item": "Focus Sash",
              "moves": [
                "Close Combat",
                "Reversal",
                "Zen Headbutt",
                "Thunder Punch"
              ],
              "ability": "Pure Power"
            },
            {
              "pokemon": "Cinderace",
              "item": "Life Orb",
              "moves": [
                "Pyro Ball",
                "Zen Headbutt",
                "Iron Head",
                "Sucker Punch"
              ],
              "ability": "Libero"
            }
          ]
        },
        {
          "trainer": "Gentleman Clifford [Double Battle With Psychic Macey]",
          "pokemon_list": [
            {
              "pokemon": "Passimian",
              "item": "Focus Sash",
              "moves": [
                "Close Combat",
                "Rock Slide",
                "Poison Jab",
                "Detect"
              ],
              "ability": "Receiver"
            },
            {
              "pokemon": "Greninja",
              "item": "Expert Belt",
              "moves": [
                "Hydro Pump",
                "Dark Pulse",
                "Ice Beam",
                "Low Kick"
              ],
              "ability": "Protean"
            },
            {
              "pokemon": "Lycanroc_Dusk",
              "item": "Life Orb",
              "moves": [
                "Stone Edge",
                "Accelerock",
                "Close Combat",
                "Psychic Fangs"
              ],
              "ability": "Tough Claws"
            },
            {
              "pokemon": "Marill",
              "item": "Quick Claw",
              "moves": [
                "Misty Explosion"
              ],
              "ability": "Huge Power"
            },
            {
              "pokemon": "Tauros",
              "item": "Dark Gem",
              "moves": [
                "Return",
                "Zen Headbutt",
                "Lash Out",
                "Role Play"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Golduck",
              "item": "Life Orb",
              "moves": [
                "Aqua Jet",
                "Cross Chop",
                "Zen Headbutt",
                "Role Play"
              ],
              "ability": "Swift Swim"
            }
          ]
        },
        {
          "trainer": "Leader Tate [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Azelf",
              "item": "Focus Sash",
              "moves": [
                "Future Sight",
                "Explosion",
                "Knock Off",
                "Stealth Rock"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Latios",
              "item": "Latiosite",
              "moves": [
                "Draco Meteor",
                "Zen Headbutt",
                "Earthquake",
                "Dragon Dance"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Zoroark",
              "item": "Dark Gem",
              "moves": [
                "Night Daze",
                "Pursuit",
                "Sludge Bomb",
                "Aura Sphere"
              ],
              "ability": "Illusion"
            },
            {
              "pokemon": "Hoopa",
              "item": "Leftovers",
              "moves": [
                "Hyperspace Hole",
                "Shadow Ball",
                "Aura Sphere",
                "Substitute"
              ],
              "ability": "Magician"
            }
          ]
        },
        {
          "trainer": "Leader Liza [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Tapu_Lele",
              "item": "Assault Vest",
              "moves": [
                "Moonblast",
                "Psychic",
                "Draining Kiss",
                "Natures Madness"
              ],
              "ability": "Psychic Surge"
            },
            {
              "pokemon": "Latias",
              "item": "Soul Dew",
              "moves": [
                "Dragon Pulse",
                "Aura Sphere",
                "Mystical Fire",
                "Calm Mind"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Metagross",
              "item": "Metagrossite",
              "moves": [
                "Meteor Mash",
                "Zen Headbutt",
                "Earthquake",
                "Pursuit"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Hoopa_Unbound",
              "item": "Focus Sash",
              "moves": [
                "Hyperspace Fury",
                "Zen Headbutt",
                "Expanding Force",
                "Aura Sphere"
              ],
              "ability": "Magician"
            }
          ]
        }
      ]
    },
    "split_name": "TnL"
  },
  "Juan": {
    "Mossdeep Space Center": {
      "zone_name": "Mossdeep Space Center",
      "zone_trainers": [
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Steelix",
              "item": "Bright Powder",
              "moves": [
                "Earthquake",
                "Gyro Ball",
                "Explosion",
                "Stealth Rock"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Appletun",
              "item": "Lum Berry",
              "moves": [
                "Dragon Pulse",
                "Apple Acid",
                "Leech Seed",
                "Recover"
              ],
              "ability": "Thick Fat"
            },
            {
              "pokemon": "Zebstrika",
              "item": "Focus Band",
              "moves": [
                "Zing Zap",
                "Overheat",
                "High Horsepower",
                "Low Kick"
              ],
              "ability": "Sap Sipper"
            },
            {
              "pokemon": "Corviknight",
              "item": "Weakness Policy",
              "moves": [
                "Body Press",
                "Power Trip",
                "Iron Defense",
                "Agility"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Simisear",
              "item": "Focus Band",
              "moves": [
                "Flamethrower",
                "Hidden Power Ice",
                "Grass Knot",
                "Nasty Plot"
              ],
              "ability": "Blaze"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt Zerk",
          "pokemon_list": [
            {
              "pokemon": "Scrafty",
              "item": "Assault Vest",
              "moves": [
                "Close Combat",
                "Knock Off",
                "Stone Edge",
                "Counter"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Duraludon",
              "item": "Iapapa Berry",
              "moves": [
                "Draco Meteor",
                "Heavy Slam",
                "Body Press",
                "Stealth Rock"
              ],
              "ability": "Heavy Metal"
            },
            {
              "pokemon": "Exeggutor",
              "item": "Quick Claw",
              "moves": [
                "Psychic",
                "Grass Knot",
                "Explosion",
                "Sleep Powder"
              ],
              "ability": "Chlorophyll"
            },
            {
              "pokemon": "Aerodactyl",
              "item": "Scope Lens",
              "moves": [
                "Stone Edge",
                "Dual Wingbeat",
                "Earthquake",
                "Dragon Dance"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Darmanitan",
              "item": "Life Orb",
              "moves": [
                "Flare Blitz",
                "Earthquake",
                "Rock Slide",
                "Reversal"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Weavile",
              "item": "Bright Powder",
              "moves": [
                "Night Slash",
                "Ice Shard",
                "Low Kick",
                "Swords Dance"
              ],
              "ability": "Inner Focus"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Diggersby",
              "item": "Quick Claw",
              "moves": [
                "Double Edge",
                "Earthquake",
                "Fire Punch",
                "Knock Off"
              ],
              "ability": "Huge Power"
            },
            {
              "pokemon": "Silvally_Grass",
              "item": "Grass Memory",
              "moves": [
                "Multi Attack",
                "Flamethrower",
                "Ice Beam",
                "Imprison"
              ],
              "ability": "RKS System"
            },
            {
              "pokemon": "Ribombee",
              "item": "Focus Band",
              "moves": [
                "Moonblast",
                "Bug Buzz",
                "Hidden Power Fire",
                "Quiver Dance"
              ],
              "ability": "Shield Dust"
            },
            {
              "pokemon": "Arcanine",
              "item": "Bright Powder",
              "moves": [
                "Flare Blitz",
                "Close Combat",
                "Wild Charge",
                "Extreme Speed"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Klinklang",
              "item": "Choice Band",
              "moves": [
                "Gear Grind"
              ],
              "ability": "Clear Body"
            }
          ]
        },
        {
          "trainer": "Magma Admin Courtney",
          "pokemon_list": [
            {
              "pokemon": "Shaymin",
              "item": "Life Orb",
              "moves": [
                "Seed Flare",
                "Earth Power",
                "Hidden Power Ice",
                "Grass Whistle"
              ],
              "ability": "Natural Cure"
            },
            {
              "pokemon": "Dugtrio",
              "item": "Ground Gem",
              "moves": [
                "Earthquake",
                "Final Gambit",
                "Memento",
                "Stealth Rock"
              ],
              "ability": "Arena Trap"
            },
            {
              "pokemon": "Tyrantrum",
              "item": "Focus Sash",
              "moves": [
                "Head Smash",
                "Scale Shot",
                "Close Combat",
                "Rock Polish"
              ],
              "ability": "Rock Head"
            },
            {
              "pokemon": "Xurkitree",
              "item": "Leftovers",
              "moves": [
                "Thunderbolt",
                "Grass Knot",
                "Hypnosis",
                "Substitute"
              ],
              "ability": "Beast Boost"
            },
            {
              "pokemon": "Buzzwole",
              "item": "Assault Vest",
              "moves": [
                "Leech Life",
                "Drain Punch",
                "Stone Edge",
                "Ice Punch"
              ],
              "ability": "Beast Boost"
            },
            {
              "pokemon": "Charizard",
              "item": "Charizardite X",
              "moves": [
                "Flare Blitz",
                "Dragon Claw",
                "Dragon Dance",
                "Roost"
              ],
              "ability": "Blaze"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Armaldo",
              "item": "Bright Powder",
              "moves": [
                "Leech Life",
                "Rock Blast",
                "Earthquake",
                "Swords Dance"
              ],
              "ability": "Battle Armor"
            },
            {
              "pokemon": "Rotom_Heat",
              "item": "White Herb",
              "moves": [
                "Overheat",
                "Thunderbolt",
                "Will O Wisp",
                "Thunder Wave"
              ],
              "ability": "Levitate"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Machamp",
              "item": "Quick Claw",
              "moves": [
                "Dynamic Punch",
                "Stone Edge",
                "Knock Off",
                "Bulk Up"
              ],
              "ability": "No Guard"
            },
            {
              "pokemon": "Talonflame",
              "item": "Flying Gem",
              "moves": [
                "Flare Blitz",
                "Acrobatics",
                "Swords Dance",
                "Will O Wisp"
              ],
              "ability": "Flame Body"
            }
          ]
        },
        {
          "trainer": "Team Magma Grunt",
          "pokemon_list": [
            {
              "pokemon": "Bisharp",
              "item": "Lum Berry",
              "moves": [
                "Iron Head",
                "Pursuit",
                "Psycho Cut",
                "Low Kick"
              ],
              "ability": "Defiant"
            },
            {
              "pokemon": "Weezing_Galarian",
              "item": "Black Sludge",
              "moves": [
                "Sludge Bomb",
                "Flamethrower",
                "Protect",
                "Will O Wisp"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Guzzlord",
              "item": "Focus Band",
              "moves": [
                "Dragon Rush",
                "Knock Off",
                "Heavy Slam",
                "Earthquake"
              ],
              "ability": "Beast Boost"
            }
          ]
        },
        {
          "trainer": "Magma Leader Maxie [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Landorus",
              "item": "Life Orb",
              "moves": [
                "Earth Power",
                "Psychic",
                "Weather Ball",
                "Protect"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Heatran",
              "item": "Air Balloon",
              "moves": [
                "Magma Storm",
                "Heat Wave",
                "Flash Cannon",
                "Earth Power"
              ],
              "ability": "Flash Fire"
            },
            {
              "pokemon": "Garchomp",
              "item": "Garchompite",
              "moves": [
                "High Horsepower",
                "Scale Shot",
                "Stone Edge",
                "Protect"
              ],
              "ability": "Rough Skin"
            }
          ]
        },
        {
          "trainer": "Magma Admin Tabitha",
          "pokemon_list": [
            {
              "pokemon": "Tyranitar",
              "item": "Tyranitarite",
              "moves": [
                "Crunch",
                "Low Kick",
                "Heavy Slam",
                "Taunt"
              ],
              "ability": "Sand Stream"
            },
            {
              "pokemon": "Ferrothorn",
              "item": "Rocky Helmet",
              "moves": [
                "Power Whip",
                "Body Press",
                "Knock Off",
                "Leech Seed"
              ],
              "ability": "Iron Barbs"
            },
            {
              "pokemon": "Moltres",
              "item": "Fire Gem",
              "moves": [
                "Burn Up",
                "Hurricane",
                "Detect",
                "Tailwind"
              ],
              "ability": "Flame Body"
            }
          ]
        },
        {
          "trainer": "Pokemon Trainer Steven",
          "pokemon_list": [
            {
              "pokemon": "Dialga",
              "item": "Dragon Gem",
              "moves": [
                "Dragon Pulse",
                "Flamethrower",
                "Aura Sphere",
                "Trick Room"
              ],
              "ability": "Telepathy"
            },
            {
              "pokemon": "Metagross",
              "item": "Life Orb",
              "moves": [
                "Meteor Mash",
                "Hammer Arm",
                "Ice Punch",
                "Trick Room"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Steelix",
              "item": "Steelixite",
              "moves": [
                "High Horsepower",
                "Heavy Slam",
                "Body Press",
                "Rock Slide"
              ],
              "ability": "Sturdy"
            }
          ]
        }
      ]
    },
    "Route 124 (South)": {
      "zone_name": "Route 124 (South)",
      "zone_trainers": [
        {
          "trainer": "Sis And Bro Lila & Roy [Double]",
          "pokemon_list": [
            {
              "pokemon": "Ludicolo",
              "item": "Focus Sash",
              "moves": [
                "Energy Ball",
                "Scald",
                "Icy Wind",
                "Fake Out"
              ],
              "ability": "Own Tempo"
            },
            {
              "pokemon": "Starmie",
              "item": "Life Orb",
              "moves": [
                "Hydro Pump",
                "Blizzard",
                "Thunder",
                "Grass Knot"
              ],
              "ability": "Analytic"
            },
            {
              "pokemon": "Dragonite",
              "item": "Lum Berry",
              "moves": [
                "Dragon Rush",
                "Dual Wingbeat",
                "Aqua Tail",
                "Thunder Wave"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Stoutland",
              "item": "Normal Gem",
              "moves": [
                "Last Resort",
                "Retaliate"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Lanturn",
              "item": "Life Orb",
              "moves": [
                "Hydro Pump",
                "Thunder",
                "Ice Beam",
                "Protect"
              ],
              "ability": "Volt Absorb"
            },
            {
              "pokemon": "Blastoise",
              "item": "Lum Berry",
              "moves": [
                "Muddy Water",
                "Focus Blast",
                "Icy Wind",
                "Fake Out"
              ],
              "ability": "Torrent"
            }
          ]
        }
      ]
    },
    "Route 126": {
      "zone_name": "Route 126",
      "zone_trainers": [
        {
          "trainer": "Swimmer\u2642 Dean",
          "pokemon_list": [
            {
              "pokemon": "Crustle",
              "item": "White Herb",
              "moves": [
                "Stone Edge",
                "Leech Life",
                "Earthquake",
                "Shell Smash"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Turtonator",
              "item": "White Herb",
              "moves": [
                "Flamethrower",
                "Dragon Pulse",
                "Scorching Sands",
                "Shell Smash"
              ],
              "ability": "Shell Armor"
            },
            {
              "pokemon": "Polteageist",
              "item": "White Herb",
              "moves": [
                "Shadow Ball",
                "Giga Drain",
                "Stored Power",
                "Shell Smash"
              ],
              "ability": "Cursed Body"
            },
            {
              "pokemon": "Minior",
              "item": "Flying Gem",
              "moves": [
                "Stone Edge",
                "Acrobatics",
                "Earthquake",
                "Shell Smash"
              ],
              "ability": "Shields Down"
            },
            {
              "pokemon": "Shuckle",
              "item": "Leftovers",
              "moves": [
                "Infestation",
                "Knock Off",
                "Rest",
                "Shell Smash"
              ],
              "ability": "Contrary"
            },
            {
              "pokemon": "Smeargle",
              "item": "Focus Sash",
              "moves": [
                "Boomburst",
                "Dark Pulse",
                "Shell Smash",
                "Spore"
              ],
              "ability": "Moody"
            }
          ]
        }
      ]
    },
    "Route 126 (Optionals)": {
      "zone_name": "Route 126 (Optionals)",
      "zone_trainers": [
        {
          "trainer": "Triathlete Pablo",
          "pokemon_list": [
            {
              "pokemon": "Crustle",
              "item": "Red Card",
              "moves": [
                "Stone Edge",
                "Earthquake",
                "Shell Smash",
                "Stealth Rock"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Tyranitar",
              "item": "Choice Band",
              "moves": [
                "Stone Edge",
                "Crunch",
                "Pursuit",
                "Heavy Slam"
              ],
              "ability": "Sand Stream"
            },
            {
              "pokemon": "Sneasler",
              "item": "Life Orb",
              "moves": [
                "Close Combat",
                "Gunk Shot",
                "Knock Off",
                "Swords Dance"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Delphox",
              "item": "Focus Sash",
              "moves": [
                "Overheat",
                "Psychic",
                "Laser Focus",
                "Hypnosis"
              ],
              "ability": "Magic Guard"
            },
            {
              "pokemon": "Kingdra",
              "item": "Leftovers",
              "moves": [
                "Outrage",
                "Waterfall",
                "Facade",
                "Dragon Dance"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Aerodactyl",
              "item": "Aerodactylite",
              "moves": [
                "Stone Edge",
                "Dual Wingbeat",
                "Earthquake",
                "Hone Claws"
              ],
              "ability": "Unnerve"
            }
          ]
        }
      ]
    },
    "Route 127": {
      "zone_name": "Route 127",
      "zone_trainers": [
        {
          "trainer": "Bird Keeper Camden [Double Battle With Battle Girl Donny]",
          "pokemon_list": [
            {
              "pokemon": "Staraptor",
              "item": "Focus Sash",
              "moves": [
                "Return",
                "Dual Wingbeat",
                "Endeavor",
                "Tailwind"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Vikavolt",
              "item": "Lum Berry",
              "moves": [
                "Thunderbolt",
                "Bug Buzz",
                "Electroweb",
                "Roost"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Hydreigon",
              "item": "Dark Gem",
              "moves": [
                "Dragon Pulse",
                "Dark Pulse",
                "Aura Sphere",
                "Roost"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Lucario",
              "item": "Life Orb",
              "moves": [
                "Aura Sphere",
                "Flash Cannon",
                "Dark Pulse",
                "Detect"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Toxicroak",
              "item": "Black Sludge",
              "moves": [
                "Sludge Bomb",
                "Aura Sphere",
                "Fake Out",
                "Detect"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Infernape",
              "item": "Focus Sash",
              "moves": [
                "Close Combat",
                "Overheat",
                "Poison Jab",
                "Fake Out"
              ],
              "ability": "Blaze"
            }
          ]
        }
      ]
    },
    "Route 127 (Optionals)": {
      "zone_name": "Route 127 (Optionals)",
      "zone_trainers": [
        {
          "trainer": "Bird Keeper Aidan [Double Battle With Cool Trainer Athena]",
          "pokemon_list": [
            {
              "pokemon": "Bruxish",
              "item": "Assault Vest",
              "moves": [
                "Liquidation",
                "Psychic Fangs",
                "Icy Wind",
                "Super Fang"
              ],
              "ability": "Dazzling"
            },
            {
              "pokemon": "Samurott_Hisuian",
              "item": "Life Orb",
              "moves": [
                "Liquidation",
                "Crunch",
                "Sucker Punch",
                "Grass Knot"
              ],
              "ability": "Torrent"
            },
            {
              "pokemon": "Hitmonlee",
              "item": "Normal Gem",
              "moves": [
                "Close Combat",
                "Retaliate",
                "Acrobatics",
                "Fake Out"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Incineroar",
              "item": "Iapapa Berry",
              "moves": [
                "Flare Blitz",
                "Throat Chop",
                "Low Kick",
                "Fake Out"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Pinsir",
              "item": "Pinsirite",
              "moves": [
                "Close Combat",
                "Return",
                "Feint",
                "Swords Dance"
              ],
              "ability": "Hyper Cutter"
            },
            {
              "pokemon": "Cinderace",
              "item": "Focus Sash",
              "moves": [
                "Pyro Ball",
                "Gunk Shot",
                "Zen Headbutt",
                "Coaching"
              ],
              "ability": "Libero"
            }
          ]
        },
        {
          "trainer": "Fisherman Roger [Double Battle With Black Belt Koji]",
          "pokemon_list": [
            {
              "pokemon": "Banette",
              "item": "Banettite",
              "moves": [
                "Shadow Claw",
                "Foul Play",
                "Swagger",
                "Thunder Wave"
              ],
              "ability": "Insomnia"
            },
            {
              "pokemon": "Grimmsnarl",
              "item": "Lum Berry",
              "moves": [
                "Foul Play",
                "Fake Out",
                "Swagger",
                "Thunder Wave"
              ],
              "ability": "Prankster"
            },
            {
              "pokemon": "Spiritomb",
              "item": "Leftovers",
              "moves": [
                "Foul Play",
                "Hex",
                "Psychic",
                "Hypnosis"
              ],
              "ability": "Pressure"
            },
            {
              "pokemon": "Ursaluna",
              "item": "Leftovers",
              "moves": [
                "High Horsepower",
                "Body Slam",
                "Body Press",
                "Bulk Up"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Corviknight",
              "item": "Weakness Policy",
              "moves": [
                "Dual Wingbeat",
                "Body Press",
                "Bulk Up",
                "Roost"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Haxorus",
              "item": "Dragon Gem",
              "moves": [
                "Dual Chop",
                "Earthquake",
                "Aqua Tail",
                "Swords Dance"
              ],
              "ability": "Unnerve"
            }
          ]
        }
      ]
    },
    "Route 128": {
      "zone_name": "Route 128",
      "zone_trainers": [
        {
          "trainer": "Swimmer\u2640 Carlee [Double Battle With Swimmer\u2642 Harrison]",
          "pokemon_list": [
            {
              "pokemon": "Samurott_Hisuian",
              "item": "Assault Vest",
              "moves": [
                "Hydro Pump",
                "Dark Pulse",
                "Icy Wind",
                "Grass Knot"
              ],
              "ability": "Torrent"
            },
            {
              "pokemon": "Decidueye_Hisuian",
              "item": "Fighting Gem",
              "moves": [
                "Close Combat",
                "Leaf Blade",
                "Knock Off",
                "Tailwind"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Typhlosion_Hisuian",
              "item": "Fire Gem",
              "moves": [
                "Fire Blast",
                "Heat Wave",
                "Shadow Ball",
                "Protect"
              ],
              "ability": "Blaze"
            },
            {
              "pokemon": "Alakazam",
              "item": "Focus Sash",
              "moves": [
                "Psychic",
                "Focus Blast",
                "Light Screen",
                "Reflect"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Lapras",
              "item": "Leftovers",
              "moves": [
                "Blizzard",
                "Scald",
                "Freeze Dry",
                "Sheer Cold"
              ],
              "ability": "Water Absorb"
            },
            {
              "pokemon": "Volcarona",
              "item": "Charti Berry",
              "moves": [
                "Heat Wave",
                "Bug Buzz",
                "Giga Drain",
                "Quiver Dance"
              ],
              "ability": "Flame Body"
            }
          ]
        }
      ]
    },
    "Route 128 (Optionals)": {
      "zone_name": "Route 128 (Optionals)",
      "zone_trainers": []
    },
    "Seafloor Cavern, permanent Aurora Veil": {
      "zone_name": "Seafloor Cavern, permanent Aurora Veil",
      "zone_trainers": [
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Kingdra",
              "item": "Scope Lens",
              "moves": [
                "Draco Meteor",
                "Octazooka",
                "Agility",
                "Focus Energy"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Exeggutor_Alolan",
              "item": "Quick Claw",
              "moves": [
                "Wood Hammer",
                "Explosion",
                "Knock Off",
                "Low Kick"
              ],
              "ability": "Harvest"
            },
            {
              "pokemon": "Honchkrow",
              "item": "Scope Lens",
              "moves": [
                "Brave Bird",
                "Night Slash",
                "Superpower",
                "Substitute"
              ],
              "ability": "Super Luck"
            },
            {
              "pokemon": "Magnezone",
              "item": "Assault Vest",
              "moves": [
                "Thunder",
                "Flash Cannon",
                "Hidden Power Grass",
                "Mirror Coat"
              ],
              "ability": "Magnet Pull"
            },
            {
              "pokemon": "Crobat",
              "item": "Life Orb",
              "moves": [
                "Sludge Wave",
                "Air Slash",
                "Giga Drain",
                "Nasty Plot"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Gorebyss",
              "item": "White Herb",
              "moves": [
                "Surf",
                "Ice Beam",
                "Psychic",
                "Shell Smash"
              ],
              "ability": "Swift Swim"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Swampert",
              "item": "Rindo Berry",
              "moves": [
                "Earthquake",
                "Liquidation",
                "Avalanche",
                "Stealth Rock"
              ],
              "ability": "Torrent"
            },
            {
              "pokemon": "Gyarados",
              "item": "Wacan Berry",
              "moves": [
                "Waterfall",
                "Earthquake",
                "Ice Fang",
                "Dragon Dance"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Haxorus",
              "item": "Lum Berry",
              "moves": [
                "Outrage",
                "Iron Tail",
                "Aqua Tail",
                "Dragon Dance"
              ],
              "ability": "Mold Breaker"
            },
            {
              "pokemon": "Muk_Alolan",
              "item": "Bright Powder",
              "moves": [
                "Poison Jab",
                "Knock Off",
                "Explosion",
                "Focus Punch"
              ],
              "ability": "Poison Touch"
            },
            {
              "pokemon": "Roserade",
              "item": "Bright Powder",
              "moves": [
                "Energy Ball",
                "Sludge Bomb",
                "Hidden Power Fire",
                "Sleep Powder"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Bruxish",
              "item": "Life Orb",
              "moves": [
                "Aqua Tail",
                "Psychic Fangs",
                "Aqua Jet",
                "Crunch"
              ],
              "ability": "Dazzling"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Cloyster",
              "item": "Bright Powder",
              "moves": [
                "Hydro Pump",
                "Explosion",
                "Toxic Spikes",
                "Shell Smash"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Malamar",
              "item": "Assault Vest",
              "moves": [
                "Night Slash",
                "Psycho Cut",
                "Superpower",
                "Facade"
              ],
              "ability": "Contrary"
            },
            {
              "pokemon": "Aerodactyl",
              "item": "Choice Band",
              "moves": [
                "Stone Edge",
                "Dual Wingbeat",
                "Aqua Tail",
                "Pursuit"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Froslass",
              "item": "Bright Powder",
              "moves": [
                "Freeze Dry",
                "Hex",
                "Nasty Plot",
                "Sing"
              ],
              "ability": "Cursed Body"
            },
            {
              "pokemon": "Crawdaunt",
              "item": "Life Orb",
              "moves": [
                "Crabhammer",
                "Knock Off",
                "Aqua Jet",
                "Swords Dance"
              ],
              "ability": "Adaptability"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Scolipede",
              "item": "Lum Berry",
              "moves": [
                "Megahorn",
                "Aqua Tail",
                "Baton Pass",
                "Swords Dance"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Drednaw",
              "item": "Leftovers",
              "moves": [
                "Liquidation",
                "Stone Edge",
                "Jaw Lock",
                "Stealth Rock"
              ],
              "ability": "Strong Jaw"
            },
            {
              "pokemon": "Dhelmise",
              "item": "Quick Claw",
              "moves": [
                "Power Whip",
                "Poltergeist",
                "Anchor Shot",
                "Synthesis"
              ],
              "ability": "Steelworker"
            },
            {
              "pokemon": "Huntail",
              "item": "White Herb",
              "moves": [
                "Liquidation",
                "Crunch",
                "Ice Fang",
                "Shell Smash"
              ],
              "ability": "Water Veil"
            }
          ]
        },
        {
          "trainer": "Team Aqua Grunt",
          "pokemon_list": [
            {
              "pokemon": "Ferrothorn",
              "item": "Quick Claw",
              "moves": [
                "Gyro Ball",
                "Explosion",
                "Spikes",
                "Stealth Rock"
              ],
              "ability": "Iron Barbs"
            },
            {
              "pokemon": "Sharpedo",
              "item": "Bright Powder",
              "moves": [
                "Hydro Pump",
                "Dark Pulse",
                "Psychic Fangs",
                "Protect"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Metagross",
              "item": "Weakness Policy",
              "moves": [
                "Meteor Mash",
                "Earthquake",
                "Thunder Punch",
                "Agility"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Zoroark",
              "item": "Lum Berry",
              "moves": [
                "Night Daze",
                "Aura Sphere",
                "Extrasensory",
                "Nasty Plot"
              ],
              "ability": "Illusion"
            },
            {
              "pokemon": "Salamence",
              "item": "Life Orb",
              "moves": [
                "Draco Meteor",
                "Dual Wingbeat",
                "Earthquake",
                "Dragon Dance"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Toxicroak",
              "item": "Focus Sash",
              "moves": [
                "Sludge Bomb",
                "Vacuum Wave",
                "Dark Pulse",
                "Nasty Plot"
              ],
              "ability": "Dry Skin"
            }
          ]
        },
        {
          "trainer": "Aqua Admin Shelly",
          "pokemon_list": [
            {
              "pokemon": "Tapu_Fini",
              "item": "Leftovers",
              "moves": [
                "Surf",
                "Moonblast",
                "Natures Madness",
                "Taunt"
              ],
              "ability": "Misty Surge"
            },
            {
              "pokemon": "Hawlucha",
              "item": "Misty Seed",
              "moves": [
                "Close Combat",
                "Acrobatics",
                "Stone Edge",
                "Swords Dance"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Gengar",
              "item": "Focus Sash",
              "moves": [
                "Sludge Wave",
                "Shadow Ball",
                "Destiny Bond",
                "Mean Look"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Darmanitan_Galarian",
              "item": "Choice Band",
              "moves": [
                "Icicle Crash",
                "U Turn"
              ],
              "ability": "Gorilla Tactics"
            },
            {
              "pokemon": "Kyurem",
              "item": "Leftovers",
              "moves": [
                "Ice Beam",
                "Freeze Dry",
                "Earth Power",
                "Roost"
              ],
              "ability": "Pressure"
            },
            {
              "pokemon": "Slowbro",
              "item": "Slowbronite",
              "moves": [
                "Psychic",
                "Surf",
                "Calm Mind",
                "Slack Off"
              ],
              "ability": "Oblivious"
            }
          ]
        },
        {
          "trainer": "Aqua Leader Archie [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Kyogre",
              "item": "Custap Berry",
              "moves": [
                "Origin Pulse",
                "Liquidation",
                "Thunder",
                "Ice Beam"
              ],
              "ability": "Drizzle"
            },
            {
              "pokemon": "Overqwil",
              "item": "Life Orb",
              "moves": [
                "Gunk Shot",
                "Throat Chop",
                "Waterfall",
                "Swords Dance"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Serperior",
              "item": "Leftovers",
              "moves": [
                "Leaf Storm",
                "Glare",
                "Leech Seed",
                "Substitute"
              ],
              "ability": "Contrary"
            },
            {
              "pokemon": "Aegislash",
              "item": "Weakness Policy",
              "moves": [
                "Flash Cannon",
                "Shadow Ball",
                "Autotomize",
                "Kings Shield"
              ],
              "ability": "Stance Change"
            },
            {
              "pokemon": "Zapdos",
              "item": "Iapapa Berry",
              "moves": [
                "Hurricane",
                "Thunder",
                "Weather Ball",
                "Roost"
              ],
              "ability": "Static"
            },
            {
              "pokemon": "Swampert",
              "item": "Swampertite",
              "moves": [
                "Earthquake",
                "Liquidation",
                "Ice Punch",
                "Power Up Punch"
              ],
              "ability": "Damp"
            }
          ]
        }
      ]
    },
    "Route 129, erratic weather": {
      "zone_name": "Route 129, erratic weather",
      "zone_trainers": [
        {
          "trainer": "Swimmer\u2642 Reed",
          "pokemon_list": [
            {
              "pokemon": "Stunfisk",
              "item": "Shuca Berry",
              "moves": [
                "Earth Power",
                "Discharge",
                "Foul Play",
                "Stealth Rock"
              ],
              "ability": "Static"
            },
            {
              "pokemon": "Greninja",
              "item": "Focus Sash",
              "moves": [
                "Dark Pulse",
                "Gunk Shot",
                "Hidden Power Fire",
                "Spikes"
              ],
              "ability": "Protean"
            },
            {
              "pokemon": "Cloyster",
              "item": "White Herb",
              "moves": [
                "Icicle Spear",
                "Weather Ball",
                "Rock Blast",
                "Shell Smash"
              ],
              "ability": "Skill Link"
            },
            {
              "pokemon": "Archeops",
              "item": "Power Herb",
              "moves": [
                "Meteor Beam",
                "Air Slash",
                "Heat Wave",
                "Endeavor"
              ],
              "ability": "Defeatist"
            },
            {
              "pokemon": "Arcanine",
              "item": "Leftovers",
              "moves": [
                "Flamethrower",
                "Solar Beam",
                "Scorching Sands",
                "Roar"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Jolteon",
              "item": "Life Orb",
              "moves": [
                "Thunderbolt",
                "Volt Switch",
                "Hidden Power Grass",
                "Weather Ball"
              ],
              "ability": "Volt Absorb"
            }
          ]
        },
        {
          "trainer": "Triathlete Chase [Double Battle With Triathlete Allison]",
          "pokemon_list": [
            {
              "pokemon": "Coalossal",
              "item": "Weakness Policy",
              "moves": [
                "Burn Up",
                "Stone Edge",
                "Heat Wave",
                "Solar Beam"
              ],
              "ability": "Steam Engine"
            },
            {
              "pokemon": "Heliolisk",
              "item": "Life Orb",
              "moves": [
                "Thunderbolt",
                "Dragon Pulse",
                "Weather Ball",
                "Grass Knot"
              ],
              "ability": "Solar Power"
            },
            {
              "pokemon": "Lapras",
              "item": "Ice Gem",
              "moves": [
                "Ice Beam",
                "Freeze Dry",
                "Solar Beam",
                "Weather Ball"
              ],
              "ability": "Shell Armor"
            },
            {
              "pokemon": "Bruxish",
              "item": "Light Clay",
              "moves": [
                "Psychic Fangs",
                "Aqua Jet",
                "Light Screen",
                "Reflect"
              ],
              "ability": "Dazzling"
            },
            {
              "pokemon": "Toucannon",
              "item": "Life Orb",
              "moves": [
                "Brave Bird",
                "Heat Wave",
                "Bullet Seed",
                "Tailwind"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Nidoking",
              "item": "Focus Sash",
              "moves": [
                "Sludge Bomb",
                "Earth Power",
                "Flamethrower",
                "Thunderbolt"
              ],
              "ability": "Sheer Force"
            }
          ]
        },
        {
          "trainer": "Swimmer\u2640 Tisha",
          "pokemon_list": [
            {
              "pokemon": "Copperajah",
              "item": "Focus Sash",
              "moves": [
                "Heavy Slam",
                "Power Whip",
                "Earthquake",
                "Stealth Rock"
              ],
              "ability": "Heavy Metal"
            },
            {
              "pokemon": "Vaporeon",
              "item": "Leftovers",
              "moves": [
                "Scald",
                "Acid Armor",
                "Baton Pass",
                "Rest"
              ],
              "ability": "Hydration"
            },
            {
              "pokemon": "Comfey",
              "item": "Pixie Plate",
              "moves": [
                "Dazzling Gleam",
                "Draining Kiss",
                "Calm Mind",
                "Leech Seed"
              ],
              "ability": "Triage"
            },
            {
              "pokemon": "Falinks",
              "item": "Lum Berry",
              "moves": [
                "Close Combat",
                "Poison Jab",
                "Throat Chop",
                "No Retreat"
              ],
              "ability": "Defiant"
            },
            {
              "pokemon": "Beartic",
              "item": "Life Orb",
              "moves": [
                "Icicle Crash",
                "Liquidation",
                "Body Press",
                "Throat Chop"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Poliwrath",
              "item": "Sitrus Berry",
              "moves": [
                "Liquidation",
                "Drain Punch",
                "Belly Drum",
                "Hypnosis"
              ],
              "ability": "Swift Swim"
            }
          ]
        },
        {
          "trainer": "Swimmer\u2642 Clarence",
          "pokemon_list": [
            {
              "pokemon": "Dragonite",
              "item": "Yache Berry",
              "moves": [
                "Dragon Claw",
                "Hurricane",
                "Hydro Pump",
                "Thunder"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Magnezone",
              "item": "Custap Berry",
              "moves": [
                "Thunder",
                "Flash Cannon",
                "Explosion",
                "Hidden Power Grass"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Jynx",
              "item": "Focus Sash",
              "moves": [
                "Psychic",
                "Freeze Dry",
                "Aura Sphere",
                "Nasty Plot"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Armaldo",
              "item": "Lum Berry",
              "moves": [
                "Stone Edge",
                "Leech Life",
                "Earthquake",
                "Swords Dance"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Tentacruel",
              "item": "Black Sludge",
              "moves": [
                "Scald",
                "Protect",
                "Substitute",
                "Toxic"
              ],
              "ability": "Rain Dish"
            },
            {
              "pokemon": "Seismitoad",
              "item": "Life Orb",
              "moves": [
                "Earth Power",
                "Power Whip",
                "Sludge Bomb",
                "Weather Ball"
              ],
              "ability": "Swift Swim"
            }
          ]
        }
      ]
    },
    "Route 130, rain": {
      "zone_name": "Route 130, rain",
      "zone_trainers": [
        {
          "trainer": "Swimmer\u2642 Rodney",
          "pokemon_list": [
            {
              "pokemon": "Mantine",
              "item": "Water Gem",
              "moves": [
                "Hydro Pump",
                "Hurricane",
                "Ice Beam",
                "Roost"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Gourgeist_Small",
              "item": "Lum Berry",
              "moves": [
                "Power Whip",
                "Poltergeist",
                "Shadow Sneak",
                "Explosion"
              ],
              "ability": "Insomnia"
            },
            {
              "pokemon": "Drifblim",
              "item": "Focus Sash",
              "moves": [
                "Shadow Ball",
                "Explosion",
                "Weather Ball",
                "Thunder Wave"
              ],
              "ability": "Aftermath"
            },
            {
              "pokemon": "Masquerain",
              "item": "Life Orb",
              "moves": [
                "Bug Buzz",
                "Air Slash",
                "Hydro Pump",
                "Stun Spore"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Blastoise",
              "item": "Leftovers",
              "moves": [
                "Scald",
                "Body Press",
                "Iron Defense",
                "Protect"
              ],
              "ability": "Rain Dish"
            },
            {
              "pokemon": "Pyukumuku",
              "item": "Leftovers",
              "moves": [
                "Double Team",
                "Recover",
                "Soak",
                "Toxic"
              ],
              "ability": "Innards Out"
            }
          ]
        },
        {
          "trainer": "Swimmer\u2640 Katie",
          "pokemon_list": [
            {
              "pokemon": "Kingdra",
              "item": "Dragon Gem",
              "moves": [
                "Hydro Pump",
                "Dragon Pulse",
                "Hurricane",
                "Ice Beam"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Rotom_Mow",
              "item": "Iapapa Berry",
              "moves": [
                "Thunder",
                "Leaf Storm",
                "Signal Beam",
                "Nasty Plot"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Toxicroak",
              "item": "Life Orb",
              "moves": [
                "Gunk Shot",
                "Cross Chop",
                "Sucker Punch",
                "Swords Dance"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Goodra",
              "item": "Leftovers",
              "moves": [
                "Breaking Swipe",
                "Aqua Tail",
                "Curse",
                "Rest"
              ],
              "ability": "Hydration"
            },
            {
              "pokemon": "Durant",
              "item": "Bug Gem",
              "moves": [
                "First Impression",
                "Iron Head",
                "Leech Life",
                "Rock Slide"
              ],
              "ability": "Hustle"
            },
            {
              "pokemon": "Drednaw",
              "item": "Choice Band",
              "moves": [
                "Head Smash",
                "Liquidation"
              ],
              "ability": "Swift Swim"
            }
          ]
        }
      ]
    },
    "Route 130 (Optionals)": {
      "zone_name": "Route 130 (Optionals)",
      "zone_trainers": []
    },
    "Route 131": {
      "zone_name": "Route 131",
      "zone_trainers": [
        {
          "trainer": "Swimmer\u2642 Zappator",
          "pokemon_list": [
            {
              "pokemon": "Overqwil",
              "item": "Lum Berry",
              "moves": [
                "Throat Chop",
                "Explosion",
                "Spikes",
                "Toxic Spikes"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Talonflame",
              "item": "Flying Gem",
              "moves": [
                "Flare Blitz",
                "Acrobatics",
                "Solar Beam",
                "Roost"
              ],
              "ability": "Flame Body"
            },
            {
              "pokemon": "Slurpuff",
              "item": "Fairy Gem",
              "moves": [
                "Dazzling Gleam",
                "Flamethrower",
                "Psychic",
                "Calm Mind"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Venusaur",
              "item": "Focus Sash",
              "moves": [
                "Solar Beam",
                "Earth Power",
                "Weather Ball",
                "Growth"
              ],
              "ability": "Chlorophyll"
            },
            {
              "pokemon": "Electivire",
              "item": "Life Orb",
              "moves": [
                "Wild Charge",
                "Earthquake",
                "Ice Punch",
                "Weather Ball"
              ],
              "ability": "Vital Spirit"
            },
            {
              "pokemon": "Slowking",
              "item": "Colbur Berry",
              "moves": [
                "Psychic",
                "Flamethrower",
                "Ice Beam",
                "Grass Knot"
              ],
              "ability": "Regenerator"
            }
          ]
        },
        {
          "trainer": "Triathlete Xayah",
          "pokemon_list": [
            {
              "pokemon": "Komala",
              "item": "Chople Berry",
              "moves": [
                "Return",
                "Wood Hammer",
                "Earthquake",
                "Play Rough"
              ],
              "ability": "Comatose"
            },
            {
              "pokemon": "Charizard",
              "item": "Charcoal",
              "moves": [
                "Flamethrower",
                "Flame Charge",
                "Solar Beam",
                "Roost"
              ],
              "ability": "Solar Power"
            },
            {
              "pokemon": "Silvally_Ghost",
              "item": "Ghost Memory",
              "moves": [
                "Multi Attack",
                "Explosion",
                "Flamethrower",
                "Grass Pledge"
              ],
              "ability": "RKS System"
            },
            {
              "pokemon": "Gliscor",
              "item": "Flying Gem",
              "moves": [
                "Earthquake",
                "Acrobatics",
                "Roost",
                "Swords Dance"
              ],
              "ability": "Hyper Cutter"
            },
            {
              "pokemon": "Gastrodon",
              "item": "Rindo Berry",
              "moves": [
                "Earth Power",
                "Weather Ball",
                "Recover",
                "Toxic"
              ],
              "ability": "Sticky Hold"
            },
            {
              "pokemon": "Primarina",
              "item": "Leftovers",
              "moves": [
                "Moonblast",
                "Psychic",
                "Weather Ball",
                "Calm Mind"
              ],
              "ability": "Torrent"
            }
          ]
        },
        {
          "trainer": "Sis And Bro Reli & Ian [Double]",
          "pokemon_list": [
            {
              "pokemon": "Victreebel",
              "item": "Focus Sash",
              "moves": [
                "Solar Blade",
                "Gunk Shot",
                "Weather Ball",
                "Sleep Powder"
              ],
              "ability": "Chlorophyll"
            },
            {
              "pokemon": "Aerodactyl",
              "item": "Rock Gem",
              "moves": [
                "Stone Edge",
                "Dual Wingbeat",
                "Protect",
                "Stealth Rock"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Cinderace",
              "item": "Iapapa Berry",
              "moves": [
                "Pyro Ball",
                "Low Sweep",
                "Super Fang",
                "Coaching"
              ],
              "ability": "Libero"
            },
            {
              "pokemon": "Mimikyu",
              "item": "Fairy Gem",
              "moves": [
                "Play Rough",
                "Shadow Claw",
                "Shadow Sneak",
                "Swords Dance"
              ],
              "ability": "Disguise"
            },
            {
              "pokemon": "Ursaluna",
              "item": "Assault Vest",
              "moves": [
                "Return",
                "High Horsepower",
                "Stone Edge",
                "Payback"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Sharpedo",
              "item": "Life Orb",
              "moves": [
                "Crunch",
                "Close Combat",
                "Psychic Fangs",
                "Protect"
              ],
              "ability": "Speed Boost"
            }
          ]
        },
        {
          "trainer": "Swimmer\u2642 Herman",
          "pokemon_list": [
            {
              "pokemon": "Araquanid",
              "item": "Coba Berry",
              "moves": [
                "Liquidation",
                "Leech Life",
                "Infestation",
                "Sticky Web"
              ],
              "ability": "Water Bubble"
            },
            {
              "pokemon": "Eelektross",
              "item": "Assault Vest",
              "moves": [
                "Thunder",
                "Aqua Tail",
                "Sludge Bomb",
                "Super Fang"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Scrafty",
              "item": "Life Orb",
              "moves": [
                "Close Combat",
                "Crunch",
                "Iron Tail",
                "Dragon Dance"
              ],
              "ability": "Moxie"
            },
            {
              "pokemon": "Metagross",
              "item": "Iapapa Berry",
              "moves": [
                "Meteor Mash",
                "Earthquake",
                "Thunder Punch",
                "Power Up Punch"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Porygon_Z",
              "item": "Chople Berry",
              "moves": [
                "Tri Attack",
                "Thunder",
                "Dark Pulse",
                "Nasty Plot"
              ],
              "ability": "Analytic"
            },
            {
              "pokemon": "Jellicent",
              "item": "Mystic Water",
              "moves": [
                "Water Spout",
                "Scald",
                "Shadow Ball",
                "Strength Sap"
              ],
              "ability": "Water Absorb"
            }
          ]
        }
      ]
    },
    "Route 131 (Optionals)": {
      "zone_name": "Route 131 (Optionals)",
      "zone_trainers": []
    },
    "Route 132 (Optionals)": {
      "zone_name": "Route 132 (Optionals)",
      "zone_trainers": []
    },
    "Route 133 (Optionals)": {
      "zone_name": "Route 133 (Optionals)",
      "zone_trainers": [
        {
          "trainer": "Expert Mollie [Double Battle With Expert Conor]",
          "pokemon_list": [
            {
              "pokemon": "Gyarados",
              "item": "Power Herb",
              "moves": [
                "Bounce",
                "Waterfall",
                "Power Whip",
                "Icy Wind"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Weezing_Galarian",
              "item": "Black Sludge",
              "moves": [
                "Sludge Bomb",
                "Strange Steam",
                "Fire Blast",
                "Will O Wisp"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Braviary",
              "item": "Iapapa Berry",
              "moves": [
                "Double Edge",
                "Dual Wingbeat",
                "Close Combat",
                "Tailwind"
              ],
              "ability": "Defiant"
            },
            {
              "pokemon": "Manectric",
              "item": "Life Orb",
              "moves": [
                "Thunderbolt",
                "Flamethrower",
                "Hidden Power Grass",
                "Protect"
              ],
              "ability": "Lightning Rod"
            },
            {
              "pokemon": "Dugtrio",
              "item": "Focus Sash",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Pursuit",
                "Reversal"
              ],
              "ability": "Arena Trap"
            },
            {
              "pokemon": "Gyarados",
              "item": "Gyaradosite",
              "moves": [
                "Crunch",
                "Waterfall",
                "Dragon Dance",
                "Protect"
              ],
              "ability": "Intimidate"
            }
          ]
        }
      ]
    },
    "Route 134 (Optionals)": {
      "zone_name": "Route 134 (Optionals)",
      "zone_trainers": []
    },
    "Sootopolis Gym (1F)": {
      "zone_name": "Sootopolis Gym (1F)",
      "zone_trainers": [
        {
          "trainer": "Lass Andrea [Double Battle With Beauty Connie]",
          "pokemon_list": [
            {
              "pokemon": "Inteleon",
              "item": "Light Clay",
              "moves": [
                "Muddy Water",
                "Icy Wind",
                "Light Screen",
                "Reflect"
              ],
              "ability": "Torrent"
            },
            {
              "pokemon": "Kangaskhan",
              "item": "Weakness Policy",
              "moves": [
                "Body Slam",
                "Aqua Tail",
                "Icy Wind",
                "Fake Out"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Decidueye_Hisuian",
              "item": "White Herb",
              "moves": [
                "Leaf Storm",
                "Close Combat",
                "Leaf Blade",
                "Knock Off"
              ],
              "ability": "Overgrow"
            },
            {
              "pokemon": "Tsareena",
              "item": "Sitrus Berry",
              "moves": [
                "Power Whip",
                "Triple Axel",
                "Acupressure",
                "Protect"
              ],
              "ability": "Queenly Majesty"
            },
            {
              "pokemon": "Toxicroak",
              "item": "Focus Sash",
              "moves": [
                "Focus Blast",
                "Gunk Shot",
                "Stone Edge",
                "Acupressure"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Milotic",
              "item": "Lum Berry",
              "moves": [
                "Muddy Water",
                "Icy Wind",
                "Hypnosis",
                "Recover"
              ],
              "ability": "Competitive"
            }
          ]
        },
        {
          "trainer": "Beauty Bridget",
          "pokemon_list": [
            {
              "pokemon": "Clefable",
              "item": "Life Orb",
              "moves": [
                "Blizzard",
                "Fire Blast",
                "Psychic",
                "Stealth Rock"
              ],
              "ability": "Magic Guard"
            },
            {
              "pokemon": "Greninja",
              "item": "Lum Berry",
              "moves": [
                "Blizzard",
                "Low Kick",
                "Spikes",
                "Toxic Spikes"
              ],
              "ability": "Protean"
            },
            {
              "pokemon": "Sandslash_Alolan",
              "item": "Life Orb",
              "moves": [
                "Iron Head",
                "Triple Axel",
                "Earthquake",
                "Stealth Rock"
              ],
              "ability": "Slush Rush"
            },
            {
              "pokemon": "Floatzel",
              "item": "Expert Belt",
              "moves": [
                "Hydro Pump",
                "Blizzard",
                "Focus Blast",
                "Taunt"
              ],
              "ability": "Water Veil"
            },
            {
              "pokemon": "Goodra_Hisuian",
              "item": "Leftovers",
              "moves": [
                "Dragon Tail",
                "Heavy Slam",
                "Earthquake",
                "Curse"
              ],
              "ability": "Shell Armor"
            }
          ]
        },
        {
          "trainer": "Lady Daphne",
          "pokemon_list": [
            {
              "pokemon": "Ribombee",
              "item": "Focus Sash",
              "moves": [
                "Moonblast",
                "Bug Buzz",
                "Psychic",
                "Sticky Web"
              ],
              "ability": "Shield Dust"
            },
            {
              "pokemon": "Bruxish",
              "item": "Tanga Berry",
              "moves": [
                "Liquidation",
                "Psychic Fangs",
                "Crunch",
                "Swords Dance"
              ],
              "ability": "Strong Jaw"
            },
            {
              "pokemon": "Primarina",
              "item": "Kebia Berry",
              "moves": [
                "Hyper Voice",
                "Draining Kiss",
                "Psychic",
                "Calm Mind"
              ],
              "ability": "Liquid Voice"
            },
            {
              "pokemon": "Samurott_Hisuian",
              "item": "Chople Berry",
              "moves": [
                "Liquidation",
                "Knock Off",
                "Sacred Sword",
                "Swords Dance"
              ],
              "ability": "Shell Armor"
            },
            {
              "pokemon": "Dragapult",
              "item": "Power Herb",
              "moves": [
                "Dragon Darts",
                "Phantom Force",
                "Aqua Tail",
                "Infestation"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Roserade",
              "item": "Life Orb",
              "moves": [
                "Sludge Bomb",
                "Grass Knot",
                "Extrasensory",
                "Sleep Powder"
              ],
              "ability": "Technician"
            }
          ]
        },
        {
          "trainer": "Pok\u00e9fan Bethany",
          "pokemon_list": [
            {
              "pokemon": "Politoed",
              "item": "Eject Button",
              "moves": [
                "Hydro Pump",
                "Focus Blast",
                "Psychic",
                "Hypnosis"
              ],
              "ability": "Drizzle"
            },
            {
              "pokemon": "Copperajah",
              "item": "Iapapa Berry",
              "moves": [
                "Iron Head",
                "Earthquake",
                "Rock Slide",
                "Stealth Rock"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Seismitoad",
              "item": "Focus Sash",
              "moves": [
                "Earth Power",
                "Power Whip",
                "Sludge Bomb",
                "Weather Ball"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Overqwil",
              "item": "Lum Berry",
              "moves": [
                "Gunk Shot",
                "Throat Chop",
                "Liquidation",
                "Destiny Bond"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Electrode_Hisuian",
              "item": "Life Orb",
              "moves": [
                "Thunder",
                "Energy Ball",
                "Explosion",
                "Hidden Power Ice"
              ],
              "ability": "Static"
            },
            {
              "pokemon": "Feraligatr",
              "item": "Life Orb",
              "moves": [
                "Liquidation",
                "Crunch",
                "Ice Punch",
                "Dragon Dance"
              ],
              "ability": "Sheer Force"
            }
          ]
        },
        {
          "trainer": "Lass Crissy",
          "pokemon_list": [
            {
              "pokemon": "Pincurchin",
              "item": "Custap Berry",
              "moves": [
                "Rising Voltage",
                "Self Destruct",
                "Hydro Pump",
                "Toxic Spikes"
              ],
              "ability": "Electric Surge"
            },
            {
              "pokemon": "Silvally_Grass",
              "item": "Grass Memory",
              "moves": [
                "Multi Attack",
                "Terrain Pulse",
                "Ice Beam",
                "Work Up"
              ],
              "ability": "RKS System"
            },
            {
              "pokemon": "Hydreigon",
              "item": "Life Orb",
              "moves": [
                "Draco Meteor",
                "Earthquake",
                "Iron Tail",
                "Aqua Tail"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Raichu_Alolan",
              "item": "Focus Sash",
              "moves": [
                "Rising Voltage",
                "Surf",
                "Grass Knot",
                "Nasty Plot"
              ],
              "ability": "Surge Surfer"
            },
            {
              "pokemon": "Slowking_Galarian",
              "item": "Electric Seed",
              "moves": [
                "Psychic",
                "Scald",
                "Calm Mind",
                "Slack Off"
              ],
              "ability": "Regenerator"
            },
            {
              "pokemon": "Clawitzer",
              "item": "Leftovers",
              "moves": [
                "Water Pulse",
                "Aura Sphere",
                "Dark Pulse",
                "Terrain Pulse"
              ],
              "ability": "Mega Launcher"
            }
          ]
        },
        {
          "trainer": "Lady Brianna",
          "pokemon_list": [
            {
              "pokemon": "Kabutops",
              "item": "Focus Sash",
              "moves": [
                "Stone Edge",
                "Liquidation",
                "Metal Sound",
                "Stealth Rock"
              ],
              "ability": "Weak Armor"
            },
            {
              "pokemon": "Gothitelle",
              "item": "Psychic Gem",
              "moves": [
                "Psychic",
                "Thunderbolt",
                "Signal Beam",
                "Hypnosis"
              ],
              "ability": "Shadow Tag"
            },
            {
              "pokemon": "Accelgor",
              "item": "Life Orb",
              "moves": [
                "Bug Buzz",
                "Focus Blast",
                "Energy Ball",
                "Spikes"
              ],
              "ability": "Sticky Hold"
            },
            {
              "pokemon": "Togedemaru",
              "item": "Salac Berry",
              "moves": [
                "Iron Tail",
                "Zing Zap",
                "Flail",
                "Endeavor"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Whiscash",
              "item": "Rindo Berry",
              "moves": [
                "Earthquake",
                "Waterfall",
                "Zen Headbutt",
                "Dragon Dance"
              ],
              "ability": "Oblivious"
            }
          ]
        },
        {
          "trainer": "Lass Pearl",
          "pokemon_list": [
            {
              "pokemon": "Cryogonal",
              "item": "Light Clay",
              "moves": [
                "Blizzard",
                "Freeze Dry",
                "Explosion",
                "Aurora Veil"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Sigilyph",
              "item": "Weakness Policy",
              "moves": [
                "Air Slash",
                "Stored Power",
                "Calm Mind",
                "Roost"
              ],
              "ability": "Magic Guard"
            },
            {
              "pokemon": "Electivire",
              "item": "Expert Belt",
              "moves": [
                "Thunderbolt",
                "Focus Blast",
                "Weather Ball",
                "Magnet Rise"
              ],
              "ability": "Vital Spirit"
            },
            {
              "pokemon": "Walrein",
              "item": "Leftovers",
              "moves": [
                "Blizzard",
                "Scald",
                "Freeze Dry",
                "Substitute"
              ],
              "ability": "Ice Body"
            },
            {
              "pokemon": "Beartic",
              "item": "Ice Gem",
              "moves": [
                "Icicle Crash",
                "Liquidation",
                "Low Kick",
                "Swords Dance"
              ],
              "ability": "Slush Rush"
            }
          ]
        },
        {
          "trainer": "Beauty Tiffany [Double Battle With Beauty Olivia]",
          "pokemon_list": [
            {
              "pokemon": "Empoleon",
              "item": "Safety Goggles",
              "moves": [
                "Hydro Pump",
                "Flash Cannon",
                "Grass Knot",
                "Sing"
              ],
              "ability": "Torrent"
            },
            {
              "pokemon": "Kommo_O",
              "item": "Throat Spray",
              "moves": [
                "Clanging Scales",
                "Aura Sphere",
                "Flash Cannon",
                "Protect"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Mamoswine",
              "item": "Focus Sash",
              "moves": [
                "Blizzard",
                "High Horsepower",
                "Freeze Dry",
                "Ice Shard"
              ],
              "ability": "Oblivious"
            },
            {
              "pokemon": "Mr_Rime",
              "item": "Focus Sash",
              "moves": [
                "Blizzard",
                "Psychic",
                "Icy Wind",
                "Fake Out"
              ],
              "ability": "Ice Body"
            },
            {
              "pokemon": "Mandibuzz",
              "item": "Rocky Helmet",
              "moves": [
                "Foul Play",
                "Dark Pulse",
                "Fake Tears",
                "Tailwind"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Arctovish",
              "item": "Water Gem",
              "moves": [
                "Blizzard",
                "Fishious Rend",
                "Freeze Dry",
                "Stone Edge"
              ],
              "ability": "Water Absorb"
            }
          ]
        },
        {
          "trainer": "Leader Juan [Double] [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Sneasler",
              "item": "Focus Sash",
              "moves": [
                "Close Combat",
                "Poison Jab",
                "Icy Wind",
                "Fake Out"
              ],
              "ability": "Poison Touch"
            },
            {
              "pokemon": "Glalie",
              "item": "Glalitite",
              "moves": [
                "Blizzard",
                "Freeze Dry",
                "Return",
                "Protect"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Keldeo",
              "item": "Lum Berry",
              "moves": [
                "Hydro Pump",
                "Secret Sword",
                "Muddy Water",
                "Coaching"
              ],
              "ability": "Justified"
            },
            {
              "pokemon": "Salamence",
              "item": "Yache Berry",
              "moves": [
                "Dragon Rush",
                "Dual Wingbeat",
                "Fire Blast",
                "Tailwind"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Basculegion",
              "item": "Water Gem",
              "moves": [
                "Aqua Tail",
                "Spirit Shackle",
                "Aqua Jet",
                "Protect"
              ],
              "ability": "Adaptability"
            },
            {
              "pokemon": "Glastrier",
              "item": "Assault Vest",
              "moves": [
                "Glacial Lance",
                "High Horsepower",
                "Body Press",
                "Facade"
              ],
              "ability": "Chilling Neigh"
            }
          ]
        }
      ]
    },
    "split_name": "Juan"
  },
  "Victory Road": {
    "Victory Road (1F)": {
      "zone_name": "Victory Road (1F)",
      "zone_trainers": [
        {
          "trainer": "Pokemon Trainer Wally",
          "pokemon_list": [
            {
              "pokemon": "Vikavolt",
              "item": "Focus Sash",
              "moves": [
                "Thunderbolt",
                "Bug Buzz",
                "Energy Ball",
                "Sticky Web"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Roserade",
              "item": "Life Orb",
              "moves": [
                "Sludge Bomb",
                "Grass Knot",
                "Dazzling Gleam",
                "Hidden Power Fire"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Azumarill",
              "item": "Sitrus Berry",
              "moves": [
                "Play Rough",
                "Aqua Jet",
                "Knock Off",
                "Belly Drum"
              ],
              "ability": "Huge Power"
            },
            {
              "pokemon": "Genesect",
              "item": "Focus Sash",
              "moves": [
                "Bug Buzz",
                "Flash Cannon",
                "Explosion",
                "Flamethrower"
              ],
              "ability": "Download"
            },
            {
              "pokemon": "Obstagoon",
              "item": "Flame Orb",
              "moves": [
                "Facade",
                "Knock Off",
                "Cross Chop",
                "Bulk Up"
              ],
              "ability": "Guts"
            },
            {
              "pokemon": "Gardevoir",
              "item": "Gardevoirite",
              "moves": [
                "Psychic",
                "Hyper Voice",
                "Mystical Fire",
                "Calm Mind"
              ],
              "ability": "Synchronize"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Hope [Double Battle With Cool Trainer Albert]",
          "pokemon_list": [
            {
              "pokemon": "Mamoswine",
              "item": "Lum Berry",
              "moves": [
                "Earthquake",
                "Icicle Crash",
                "Freeze Dry",
                "Knock Off"
              ],
              "ability": "Oblivious"
            },
            {
              "pokemon": "Donphan",
              "item": "Custap Berry",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Knock Off",
                "Endeavor"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Venusaur",
              "item": "Venusaurite",
              "moves": [
                "Sludge Bomb",
                "Giga Drain",
                "Earthquake",
                "Sleep Powder"
              ],
              "ability": "Overgrow"
            },
            {
              "pokemon": "Aerodactyl",
              "item": "Aerodactylite",
              "moves": [
                "Rock Slide",
                "Dual Wingbeat",
                "Fire Fang",
                "Tailwind"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Staraptor",
              "item": "Flying Gem",
              "moves": [
                "Double Edge",
                "Acrobatics",
                "Close Combat",
                "Tailwind"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Hydreigon",
              "item": "Fire Gem",
              "moves": [
                "Dragon Pulse",
                "Dark Pulse",
                "Heat Wave",
                "Tailwind"
              ],
              "ability": "Levitate"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Katelynn",
          "pokemon_list": [
            {
              "pokemon": "Walrein",
              "item": "Lum Berry",
              "moves": [
                "Ice Beam",
                "Scald",
                "Whirlpool",
                "Super Fang"
              ],
              "ability": "Thick Fat"
            },
            {
              "pokemon": "Yanmega",
              "item": "Leftovers",
              "moves": [
                "Bug Buzz",
                "Air Slash",
                "Hypnosis",
                "Substitute"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Ludicolo",
              "item": "Eject Pack",
              "moves": [
                "Leaf Storm",
                "Scald",
                "Knock Off",
                "Icy Wind"
              ],
              "ability": "Own Tempo"
            },
            {
              "pokemon": "Dugtrio",
              "item": "Life Orb",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Night Slash",
                "Pursuit"
              ],
              "ability": "Arena Trap"
            },
            {
              "pokemon": "Magnezone",
              "item": "Air Balloon",
              "moves": [
                "Flash Cannon",
                "Discharge",
                "Explosion",
                "Hidden Power Fire"
              ],
              "ability": "Magnet Pull"
            },
            {
              "pokemon": "Pinsir",
              "item": "Pinsirite",
              "moves": [
                "Earthquake",
                "Frustration",
                "Quick Attack",
                "Swords Dance"
              ],
              "ability": "Hyper Cutter"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Quincy",
          "pokemon_list": [
            {
              "pokemon": "Carbink",
              "item": "Leftovers",
              "moves": [
                "Explosion",
                "Body Press",
                "Spikes",
                "Stealth Rock"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Slowking",
              "item": "Assault Vest",
              "moves": [
                "Future Sight",
                "Scald",
                "Flamethrower",
                "Dragon Tail"
              ],
              "ability": "Regenerator"
            },
            {
              "pokemon": "Trevenant",
              "item": "Sitrus Berry",
              "moves": [
                "Poltergeist",
                "Horn Leech",
                "Leech Seed",
                "Substitute"
              ],
              "ability": "Harvest"
            },
            {
              "pokemon": "Muk",
              "item": "Black Sludge",
              "moves": [
                "Gunk Shot",
                "Explosion",
                "Fire Punch",
                "Curse"
              ],
              "ability": "Poison Touch"
            },
            {
              "pokemon": "Kommo_O",
              "item": "Dragon Gem",
              "moves": [
                "Close Combat",
                "Dragon Claw",
                "Poison Jab",
                "Dragon Dance"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Beedrill",
              "item": "Beedrillite",
              "moves": [
                "X Scissor",
                "Drill Run",
                "Knock Off",
                "Swords Dance"
              ],
              "ability": "Swarm"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Felix",
          "pokemon_list": [
            {
              "pokemon": "Rillaboom",
              "item": "Focus Sash",
              "moves": [
                "Grassy Glide",
                "Knock Off",
                "Low Sweep",
                "Endeavor"
              ],
              "ability": "Grassy Surge"
            },
            {
              "pokemon": "Tyranitar",
              "item": "Dark Gem",
              "moves": [
                "Stone Edge",
                "Crunch",
                "Pursuit",
                "Stealth Rock"
              ],
              "ability": "Sand Stream"
            },
            {
              "pokemon": "Clefable",
              "item": "Grassy Seed",
              "moves": [
                "Moonblast",
                "Flamethrower",
                "Calm Mind",
                "Soft Boiled"
              ],
              "ability": "Magic Guard"
            },
            {
              "pokemon": "Arcanine_Hisuian",
              "item": "Lum Berry",
              "moves": [
                "Head Smash",
                "Flare Blitz",
                "Close Combat",
                "Extreme Speed"
              ],
              "ability": "Rock Head"
            },
            {
              "pokemon": "Inteleon",
              "item": "Flying Gem",
              "moves": [
                "Snipe Shot",
                "Hidden Power Grass",
                "Acrobatics",
                "Work Up"
              ],
              "ability": "Sniper"
            },
            {
              "pokemon": "Ampharos",
              "item": "Ampharosite",
              "moves": [
                "Thunderbolt",
                "Dragon Pulse",
                "Focus Blast",
                "Agility"
              ],
              "ability": "Static"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Julie [Double Battle With Cool Trainer Dianne]",
          "pokemon_list": [
            {
              "pokemon": "Sceptile",
              "item": "Sceptilite",
              "moves": [
                "Energy Ball",
                "Dragon Pulse",
                "Focus Blast",
                "Detect"
              ],
              "ability": "Overgrow"
            },
            {
              "pokemon": "Sirfetchd",
              "item": "Fighting Gem",
              "moves": [
                "Close Combat",
                "Poison Jab",
                "Knock Off",
                "Detect"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Florges",
              "item": "Salac Berry",
              "moves": [
                "Dazzling Gleam",
                "Psychic",
                "Pollen Puff",
                "Calm Mind"
              ],
              "ability": "Symbiosis"
            },
            {
              "pokemon": "Golisopod",
              "item": "Assault Vest",
              "moves": [
                "First Impression",
                "Liquidation",
                "Leech Life",
                "Knock Off"
              ],
              "ability": "Emergency Exit"
            },
            {
              "pokemon": "Honchkrow",
              "item": "Focus Sash",
              "moves": [
                "Hurricane",
                "Dark Pulse",
                "Heat Wave",
                "Tailwind"
              ],
              "ability": "Super Luck"
            },
            {
              "pokemon": "Kangaskhan",
              "item": "Kangaskhanite",
              "moves": [
                "Body Slam",
                "Crunch",
                "Seismic Toss",
                "Fake Out"
              ],
              "ability": "Scrappy"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Samuel",
          "pokemon_list": [
            {
              "pokemon": "Scolipede",
              "item": "Focus Sash",
              "moves": [
                "Infestation",
                "Endeavor",
                "Spikes",
                "Toxic Spikes"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Garchomp",
              "item": "Life Orb",
              "moves": [
                "Scale Shot",
                "Earthquake",
                "Stone Edge",
                "Stealth Rock"
              ],
              "ability": "Rough Skin"
            },
            {
              "pokemon": "Tauros",
              "item": "Life Orb",
              "moves": [
                "Body Slam",
                "Close Combat",
                "Zen Headbutt",
                "Throat Chop"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Gyarados",
              "item": "Lum Berry",
              "moves": [
                "Waterfall",
                "Power Whip",
                "Ice Fang",
                "Dragon Dance"
              ],
              "ability": "Moxie"
            },
            {
              "pokemon": "Eelektross",
              "item": "Assault Vest",
              "moves": [
                "Discharge",
                "Flamethrower",
                "Sludge Bomb",
                "Knock Off"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Blaziken",
              "item": "Blazikenite",
              "moves": [
                "Close Combat",
                "Flare Blitz",
                "Stone Edge",
                "Swords Dance"
              ],
              "ability": "Blaze"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Shannon",
          "pokemon_list": [
            {
              "pokemon": "Pelipper",
              "item": "Wacan Berry",
              "moves": [
                "Hydro Pump",
                "Hurricane",
                "Icy Wind",
                "Detect"
              ],
              "ability": "Drizzle"
            },
            {
              "pokemon": "Gorebyss",
              "item": "White Herb",
              "moves": [
                "Surf",
                "Ice Beam",
                "Baton Pass",
                "Shell Smash"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Dragonite",
              "item": "Lum Berry",
              "moves": [
                "Hurricane",
                "Dragon Claw",
                "Hydro Pump",
                "Earthquake"
              ],
              "ability": "Multiscale"
            },
            {
              "pokemon": "Scizor",
              "item": "Steel Gem",
              "moves": [
                "Bug Bite",
                "Bullet Punch",
                "Quick Attack",
                "Swords Dance"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Whimsicott",
              "item": "Life Orb",
              "moves": [
                "Energy Ball",
                "Hurricane",
                "Weather Ball",
                "Grass Whistle"
              ],
              "ability": "Prankster"
            },
            {
              "pokemon": "Swampert",
              "item": "Swampertite",
              "moves": [
                "Earthquake",
                "Liquidation",
                "Stone Edge",
                "Power Up Punch"
              ],
              "ability": "Damp"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Caroline",
          "pokemon_list": [
            {
              "pokemon": "Mudsdale",
              "item": "Assault Vest",
              "moves": [
                "Earthquake",
                "Body Press",
                "Payback",
                "Heavy Slam"
              ],
              "ability": "Stamina"
            },
            {
              "pokemon": "Ferrothorn",
              "item": "Occa Berry",
              "moves": [
                "Power Whip",
                "Gyro Ball",
                "Spikes",
                "Stealth Rock"
              ],
              "ability": "Iron Barbs"
            },
            {
              "pokemon": "Blastoise",
              "item": "White Herb",
              "moves": [
                "Scald",
                "Earthquake",
                "Ice Beam",
                "Shell Smash"
              ],
              "ability": "Torrent"
            },
            {
              "pokemon": "Zoroark_Hisuian",
              "item": "Focus Sash",
              "moves": [
                "Hyper Voice",
                "Flamethrower",
                "Grass Knot",
                "Nasty Plot"
              ],
              "ability": "Illusion"
            },
            {
              "pokemon": "Braviary_Hisuian",
              "item": "Life Orb",
              "moves": [
                "Hurricane",
                "Psychic",
                "Heat Wave",
                "Agility"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Gallade",
              "item": "Galladite",
              "moves": [
                "Close Combat",
                "Zen Headbutt",
                "Poison Jab",
                "Knock Off"
              ],
              "ability": "Inner Focus"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Michelle",
          "pokemon_list": [
            {
              "pokemon": "Slurpuff",
              "item": "Focus Sash",
              "moves": [
                "Misty Explosion",
                "Fire Blast",
                "Endeavor",
                "Sticky Web"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Darmanitan",
              "item": "Life Orb",
              "moves": [
                "Flare Blitz",
                "Earthquake",
                "Zen Headbutt",
                "Rock Slide"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Chandelure",
              "item": "Choice Specs",
              "moves": [
                "Fire Blast",
                "Shadow Ball",
                "Energy Ball",
                "Hidden Power Ice"
              ],
              "ability": "Shadow Tag"
            },
            {
              "pokemon": "Gliscor",
              "item": "Toxic Orb",
              "moves": [
                "Earthquake",
                "Facade",
                "Knock Off",
                "Stealth Rock"
              ],
              "ability": "Poison Heal"
            },
            {
              "pokemon": "Hydreigon",
              "item": "Lum Berry",
              "moves": [
                "Dragon Pulse",
                "Dark Pulse",
                "Flamethrower",
                "Nasty Plot"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Heracross",
              "item": "Heracronite",
              "moves": [
                "Close Combat",
                "Pin Missile",
                "Rock Blast",
                "Substitute"
              ],
              "ability": "Swarm"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Specter",
          "pokemon_list": [
            {
              "pokemon": "Snorlax",
              "item": "Custap Berry",
              "moves": [
                "Self Destruct",
                "Earthquake",
                "Crunch",
                "Counter"
              ],
              "ability": "Gluttony"
            },
            {
              "pokemon": "Salamence",
              "item": "Life Orb",
              "moves": [
                "Draco Meteor",
                "Dual Wingbeat",
                "Earthquake",
                "Fire Fang"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Breloom",
              "item": "Focus Sash",
              "moves": [
                "Close Combat",
                "Mach Punch",
                "Bullet Seed",
                "Rock Tomb"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Metagross",
              "item": "Weakness Policy",
              "moves": [
                "Meteor Mash",
                "Body Press",
                "Stored Power",
                "Cosmic Power"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Starmie",
              "item": "Power Herb",
              "moves": [
                "Hydro Pump",
                "Psychic",
                "Meteor Beam",
                "Ice Beam"
              ],
              "ability": "Analytic"
            },
            {
              "pokemon": "Sableye",
              "item": "Sablenite",
              "moves": [
                "Seismic Toss",
                "Hex",
                "Will O Wisp",
                "Recover"
              ],
              "ability": "Keen Eye"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Edgar",
          "pokemon_list": [
            {
              "pokemon": "Charizard",
              "item": "Charizardite Y",
              "moves": [
                "Heat Wave",
                "Air Slash",
                "Focus Blast",
                "Solar Beam"
              ],
              "ability": "Blaze"
            },
            {
              "pokemon": "Rhyperior",
              "item": "Leftovers",
              "moves": [
                "Head Smash",
                "Earthquake",
                "Heat Crash",
                "Stealth Rock"
              ],
              "ability": "Rock Head"
            },
            {
              "pokemon": "Lilligant_Hisuian",
              "item": "Flying Gem",
              "moves": [
                "Solar Blade",
                "Close Combat",
                "Acrobatics",
                "Sleep Powder"
              ],
              "ability": "Chlorophyll"
            },
            {
              "pokemon": "Ninetales_Alolan",
              "item": "Focus Sash",
              "moves": [
                "Moonblast",
                "Freeze Dry",
                "Weather Ball",
                "Nasty Plot"
              ],
              "ability": "Snow Cloak"
            },
            {
              "pokemon": "Venusaur",
              "item": "Life Orb",
              "moves": [
                "Solar Beam",
                "Earthquake",
                "Weather Ball",
                "Growth"
              ],
              "ability": "Chlorophyll"
            },
            {
              "pokemon": "Goodra",
              "item": "Assault Vest",
              "moves": [
                "Dragon Pulse",
                "Earthquake",
                "Weather Ball",
                "Counter"
              ],
              "ability": "Gooey"
            }
          ]
        },
        {
          "trainer": "Triathlete Darren",
          "pokemon_list": [
            {
              "pokemon": "Archeops",
              "item": "Flying Gem",
              "moves": [
                "Power Gem",
                "Acrobatics",
                "Heat Wave",
                "Stealth Rock"
              ],
              "ability": "Defeatist"
            },
            {
              "pokemon": "Wobbuffet",
              "item": "Iapapa Berry",
              "moves": [
                "Counter",
                "Mirror Coat",
                "Destiny Bond",
                "Encore"
              ],
              "ability": "Shadow Tag"
            },
            {
              "pokemon": "Nidoking",
              "item": "Focus Sash",
              "moves": [
                "Earth Power",
                "Sludge Wave",
                "Ice Beam",
                "Counter"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Dragapult",
              "item": "Life Orb",
              "moves": [
                "Dragon Darts",
                "Fire Blast",
                "Thunder",
                "Steel Wing"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Pikachu_World_Cap",
              "item": "Light Ball",
              "moves": [
                "Volt Tackle",
                "Thunder",
                "Surf",
                "Grass Knot"
              ],
              "ability": "Static"
            },
            {
              "pokemon": "Charizard",
              "item": "Charizardite X",
              "moves": [
                "Flare Blitz",
                "Dragon Claw",
                "Dragon Dance",
                "Roost"
              ],
              "ability": "Blaze"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Halle",
          "pokemon_list": [
            {
              "pokemon": "Golem",
              "item": "Custap Berry",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Explosion",
                "Stealth Rock"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Turtonator",
              "item": "Leftovers",
              "moves": [
                "Heat Crash",
                "Explosion",
                "Body Press",
                "Curse"
              ],
              "ability": "Shell Armor"
            },
            {
              "pokemon": "Floette_Eternal_Flower",
              "item": "Sitrus Berry",
              "moves": [
                "Light Of Ruin",
                "Psychic",
                "Hidden Power Fire",
                "Calm Mind"
              ],
              "ability": "Flower Veil"
            },
            {
              "pokemon": "Escavalier",
              "item": "Assault Vest",
              "moves": [
                "Megahorn",
                "Iron Head",
                "Drill Run",
                "Knock Off"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Weavile",
              "item": "Focus Sash",
              "moves": [
                "Knock Off",
                "Triple Axel",
                "Low Kick",
                "Swords Dance"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Mawile",
              "item": "Mawilite",
              "moves": [
                "Play Rough",
                "Iron Head",
                "Sucker Punch",
                "Swords Dance"
              ],
              "ability": "Intimidate"
            }
          ]
        },
        {
          "trainer": "Winstrate Vito [Boss]",
          "pokemon_list": [
            {
              "pokemon": "Indeedee",
              "item": "Focus Sash",
              "moves": [
                "Future Sight",
                "Expanding Force",
                "Dazzling Gleam",
                "Mystical Fire"
              ],
              "ability": "Psychic Surge"
            },
            {
              "pokemon": "Aggron",
              "item": "Custap Berry",
              "moves": [
                "Stone Edge",
                "Heavy Slam",
                "Metal Burst",
                "Stealth Rock"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Pheromosa",
              "item": "Life Orb",
              "moves": [
                "Close Combat",
                "Bug Buzz",
                "Ice Beam",
                "Throat Chop"
              ],
              "ability": "Beast Boost"
            },
            {
              "pokemon": "Cloyster",
              "item": "Focus Sash",
              "moves": [
                "Hydro Pump",
                "Freeze Dry",
                "Explosion",
                "Shell Smash"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Hawlucha",
              "item": "Psychic Seed",
              "moves": [
                "Close Combat",
                "Acrobatics",
                "Stone Edge",
                "Swords Dance"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Alakazam",
              "item": "Alakazite",
              "moves": [
                "Expanding Force",
                "Dazzling Gleam",
                "Shadow Ball",
                "Substitute"
              ],
              "ability": "Magic Guard"
            }
          ]
        }
      ]
    },
    "Route 123": {
      "zone_name": "Route 123",
      "zone_trainers": [
        {
          "trainer": "Pok\u00e9maniac Hambino",
          "pokemon_list": [
            {
              "pokemon": "Togekiss",
              "item": "Lum Berry",
              "moves": [
                "Air Slash",
                "Aura Sphere",
                "Mystical Fire",
                "Thunder Wave"
              ],
              "ability": "Serene Grace"
            },
            {
              "pokemon": "Marowak",
              "item": "Thick Club",
              "moves": [
                "Bonemerang",
                "Stone Edge",
                "Fire Punch",
                "Stealth Rock"
              ],
              "ability": "Battle Armor"
            },
            {
              "pokemon": "Serperior",
              "item": "Focus Sash",
              "moves": [
                "Leaf Storm",
                "Dragon Pulse",
                "Hidden Power Fire",
                "Glare"
              ],
              "ability": "Contrary"
            },
            {
              "pokemon": "Drifblim",
              "item": "Flying Gem",
              "moves": [
                "Shadow Ball",
                "Acrobatics",
                "Explosion",
                "Hidden Power Fire"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Hatterene",
              "item": "Babiri Berry",
              "moves": [
                "Misty Explosion",
                "Psychic",
                "Mystical Fire",
                "Nuzzle"
              ],
              "ability": "Magic Bounce"
            },
            {
              "pokemon": "Luxray",
              "item": "Flame Orb",
              "moves": [
                "Zing Zap",
                "Facade",
                "Crunch",
                "Agility"
              ],
              "ability": "Guts"
            }
          ]
        },
        {
          "trainer": "Cool Trainer Wendy [Double Battle With Cool Trainer Braxton]",
          "pokemon_list": [
            {
              "pokemon": "Oranguru",
              "item": "Starf Berry",
              "moves": [
                "Psychic",
                "Stored Power",
                "Psych Up",
                "Trick Room"
              ],
              "ability": "Symbiosis"
            },
            {
              "pokemon": "Florges",
              "item": "Starf Berry",
              "moves": [
                "Moonblast",
                "Dazzling Gleam",
                "Pollen Puff",
                "Aromatherapy"
              ],
              "ability": "Symbiosis"
            },
            {
              "pokemon": "Vivillon_Garden",
              "item": "Focus Sash",
              "moves": [
                "Pollen Puff",
                "Rage Powder",
                "Sleep Powder",
                "Stun Spore"
              ],
              "ability": "Friend Guard"
            },
            {
              "pokemon": "Greedent",
              "item": "Starf Berry",
              "moves": [
                "Return",
                "Body Press",
                "Crunch",
                "Stuff Cheeks"
              ],
              "ability": "Cheek Pouch"
            },
            {
              "pokemon": "Swalot",
              "item": "Starf Berry",
              "moves": [
                "Belch",
                "Body Press",
                "Fire Punch",
                "Stuff Cheeks"
              ],
              "ability": "Sticky Hold"
            },
            {
              "pokemon": "Snorlax",
              "item": "Starf Berry",
              "moves": [
                "Body Slam",
                "High Horsepower",
                "Crunch",
                "Curse"
              ],
              "ability": "Gluttony"
            }
          ]
        },
        {
          "trainer": "Ninja Boy Jonas [Double Battle With Expert Fredrick]",
          "pokemon_list": [
            {
              "pokemon": "Togekiss",
              "item": "Salac Berry",
              "moves": [
                "Dazzling Gleam",
                "Stored Power",
                "Fling",
                "Psych Up"
              ],
              "ability": "Super Luck"
            },
            {
              "pokemon": "Sneasler",
              "item": "Salac Berry",
              "moves": [
                "Close Combat",
                "Poison Jab",
                "Ice Shard",
                "Fling"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Donphan",
              "item": "Custap Berry",
              "moves": [
                "High Horsepower",
                "Stone Edge",
                "Ice Shard",
                "Endeavor"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Mr_Rime",
              "item": "Weakness Policy",
              "moves": [
                "Blizzard",
                "Stored Power",
                "Freeze Dry",
                "Fake Out"
              ],
              "ability": "Tangled Feet"
            },
            {
              "pokemon": "Decidueye",
              "item": "Weakness Policy",
              "moves": [
                "Energy Ball",
                "Shadow Ball",
                "Hidden Power Fire",
                "Tailwind"
              ],
              "ability": "Overgrow"
            },
            {
              "pokemon": "Sceptile",
              "item": "Weakness Policy",
              "moves": [
                "Energy Ball",
                "Focus Blast",
                "Rock Slide",
                "Acrobatics"
              ],
              "ability": "Unburden"
            }
          ]
        },
        {
          "trainer": "Hex Maniac Gongas",
          "pokemon_list": [
            {
              "pokemon": "Raichu",
              "item": "Focus Sash",
              "moves": [
                "Thunder",
                "Surf",
                "Grass Knot",
                "Counter"
              ],
              "ability": "Static"
            },
            {
              "pokemon": "Glaceon",
              "item": "Occa Berry",
              "moves": [
                "Blizzard",
                "Freeze Dry",
                "Hidden Power Ground",
                "Calm Mind"
              ],
              "ability": "Snow Cloak"
            },
            {
              "pokemon": "Swellow",
              "item": "Life Orb",
              "moves": [
                "Boomburst",
                "Brave Bird",
                "Heat Wave",
                "U Turn"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Tangrowth",
              "item": "Assault Vest",
              "moves": [
                "Power Whip",
                "Earthquake",
                "Knock Off",
                "Infestation"
              ],
              "ability": "Regenerator"
            },
            {
              "pokemon": "Volcarona",
              "item": "Grass Gem",
              "moves": [
                "Bug Buzz",
                "Fiery Dance",
                "Giga Drain",
                "Quiver Dance"
              ],
              "ability": "Flame Body"
            },
            {
              "pokemon": "Slowbro",
              "item": "Leftovers",
              "moves": [
                "Future Sight",
                "Scald",
                "Body Press",
                "Slack Off"
              ],
              "ability": "Regenerator"
            }
          ]
        },
        {
          "trainer": "Twins Miu & Yuki [Double]",
          "pokemon_list": [
            {
              "pokemon": "Sharpedo",
              "item": "Focus Sash",
              "moves": [
                "Hydro Pump",
                "Dark Pulse",
                "Icy Wind",
                "Destiny Bond"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Marowak_Alolan",
              "item": "Thick Club",
              "moves": [
                "Flare Blitz",
                "Bonemerang",
                "Protect",
                "Stealth Rock"
              ],
              "ability": "Lightning Rod"
            },
            {
              "pokemon": "Gothitelle",
              "item": "Leftovers",
              "moves": [
                "Psychic",
                "Fake Out",
                "Protect",
                "Thunder Wave"
              ],
              "ability": "Shadow Tag"
            },
            {
              "pokemon": "Granbull",
              "item": "Iapapa Berry",
              "moves": [
                "Play Rough",
                "Close Combat",
                "Stomping Tantrum",
                "Thunder Wave"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Toxicroak",
              "item": "White Herb",
              "moves": [
                "Close Combat",
                "Poison Jab",
                "Knock Off",
                "Fake Out"
              ],
              "ability": "Poison Touch"
            },
            {
              "pokemon": "Arcanine_Hisuian",
              "item": "Rock Gem",
              "moves": [
                "Flare Blitz",
                "Rock Slide",
                "Extreme Speed",
                "Howl"
              ],
              "ability": "Intimidate"
            }
          ]
        },
        {
          "trainer": "Bug Catcher Davis",
          "pokemon_list": [
            {
              "pokemon": "Kingdra",
              "item": "Life Orb",
              "moves": [
                "Dragon Pulse",
                "Octazooka",
                "Flip Turn",
                "Hurricane"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Basculegion",
              "item": "Life Orb",
              "moves": [
                "Aqua Tail",
                "Spirit Shackle",
                "Head Smash",
                "Destiny Bond"
              ],
              "ability": "Mold Breaker"
            },
            {
              "pokemon": "Exeggutor_Alolan",
              "item": "Yache Berry",
              "moves": [
                "Leaf Storm",
                "Dragon Hammer",
                "Explosion",
                "Low Kick"
              ],
              "ability": "Harvest"
            },
            {
              "pokemon": "Abomasnow",
              "item": "Ice Gem",
              "moves": [
                "Wood Hammer",
                "Ice Shard",
                "Earthquake",
                "Swords Dance"
              ],
              "ability": "Soundproof"
            },
            {
              "pokemon": "Zangoose",
              "item": "Sitrus Berry",
              "moves": [
                "Quick Attack",
                "Close Combat",
                "Night Slash",
                "Belly Drum"
              ],
              "ability": "Toxic Boost"
            },
            {
              "pokemon": "Scizor",
              "item": "Scizorite",
              "moves": [
                "Bullet Punch",
                "Close Combat",
                "Knock Off",
                "Swords Dance"
              ],
              "ability": "Swarm"
            }
          ]
        },
        {
          "trainer": "Psychic Jacki [Double Battle With Bird Keeper Alberto]",
          "pokemon_list": [
            {
              "pokemon": "Wyrdeer",
              "item": "Life Orb",
              "moves": [
                "Psychic",
                "Megahorn",
                "Thunder",
                "Earthquake"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Metagross",
              "item": "Ground Gem",
              "moves": [
                "Meteor Mash",
                "Zen Headbutt",
                "Earthquake",
                "Ice Punch"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Medicham",
              "item": "Focus Sash",
              "moves": [
                "Close Combat",
                "Zen Headbutt",
                "Reversal",
                "Thunder Punch"
              ],
              "ability": "Pure Power"
            },
            {
              "pokemon": "Pelipper",
              "item": "Water Gem",
              "moves": [
                "Hurricane",
                "Weather Ball",
                "Detect",
                "Tailwind"
              ],
              "ability": "Drizzle"
            },
            {
              "pokemon": "Tropius",
              "item": "Sitrus Berry",
              "moves": [
                "Hurricane",
                "Grass Knot",
                "Helping Hand",
                "Tailwind"
              ],
              "ability": "Harvest"
            },
            {
              "pokemon": "Charizard",
              "item": "Flying Gem",
              "moves": [
                "Burn Up",
                "Hurricane",
                "Focus Blast",
                "Weather Ball"
              ],
              "ability": "Blaze"
            }
          ]
        },
        {
          "trainer": "Black Belt Hawk",
          "pokemon_list": [
            {
              "pokemon": "Miltank",
              "item": "Chople Berry",
              "moves": [
                "Body Slam",
                "Milk Drink",
                "Stealth Rock",
                "Thunder Wave"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Hitmontop",
              "item": "Life Orb",
              "moves": [
                "Close Combat",
                "Mach Punch",
                "Bullet Punch",
                "Triple Axel"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Krookodile",
              "item": "Dark Gem",
              "moves": [
                "Earthquake",
                "Pursuit",
                "Stone Edge",
                "Aqua Tail"
              ],
              "ability": "Moxie"
            },
            {
              "pokemon": "Noivern",
              "item": "Flying Gem",
              "moves": [
                "Dragon Rush",
                "Acrobatics",
                "Heat Wave",
                "Dragon Dance"
              ],
              "ability": "Infiltrator"
            },
            {
              "pokemon": "Scizor",
              "item": "Focus Sash",
              "moves": [
                "Bug Bite",
                "Bullet Punch",
                "Close Combat",
                "Swords Dance"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Mudsdale",
              "item": "Assault Vest",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Heavy Slam",
                "Counter"
              ],
              "ability": "Inner Focus"
            }
          ]
        },
        {
          "trainer": "Guitarist Fernando [Double Battle With Cool Trainer Jazmyn]",
          "pokemon_list": [
            {
              "pokemon": "Silvally_Electric",
              "item": "Electric Memory",
              "moves": [
                "Multi Attack",
                "Explosion",
                "Grass Pledge",
                "Tailwind"
              ],
              "ability": "RKS System"
            },
            {
              "pokemon": "Kommo_O",
              "item": "Normal Gem",
              "moves": [
                "Dragon Pulse",
                "Aura Sphere",
                "Boomburst"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Golem",
              "item": "Custap Berry",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Rock Slide",
                "Explosion"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Froslass",
              "item": "Ice Gem",
              "moves": [
                "Blizzard",
                "Shadow Ball",
                "Hex",
                "Lovely Kiss"
              ],
              "ability": "Cursed Body"
            },
            {
              "pokemon": "Orbeetle",
              "item": "Iapapa Berry",
              "moves": [
                "Future Sight",
                "Bug Buzz",
                "Infestation",
                "Hypnosis"
              ],
              "ability": "Telepathy"
            },
            {
              "pokemon": "Gengar",
              "item": "Focus Sash",
              "moves": [
                "Sludge Wave",
                "Shadow Ball",
                "Focus Blast",
                "Hypnosis"
              ],
              "ability": "Levitate"
            }
          ]
        },
        {
          "trainer": "Psychic Cameron [Double Battle With Hex Maniac Kindra]",
          "pokemon_list": [
            {
              "pokemon": "Mienshao",
              "item": "Life Orb",
              "moves": [
                "Close Combat",
                "Rock Slide",
                "Knock Off",
                "Detect"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Mimikyu",
              "item": "Fairy Gem",
              "moves": [
                "Play Rough",
                "Shadow Claw",
                "Drain Punch",
                "Swords Dance"
              ],
              "ability": "Disguise"
            },
            {
              "pokemon": "Thievul",
              "item": "Psychic Seed",
              "moves": [
                "Foul Play",
                "Dark Pulse",
                "Burning Jealousy",
                "Fake Tears"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Indeedee_Female",
              "item": "Light Clay",
              "moves": [
                "Expanding Force",
                "Encore",
                "Reflect",
                "Light Screen"
              ],
              "ability": "Psychic Surge"
            },
            {
              "pokemon": "Alakazam",
              "item": "Life Orb",
              "moves": [
                "Expanding Force",
                "Aura Sphere",
                "Dazzling Gleam",
                "Protect"
              ],
              "ability": "Magic Guard"
            },
            {
              "pokemon": "Starmie",
              "item": "Focus Sash",
              "moves": [
                "Hydro Pump",
                "Expanding Force",
                "Whirlpool",
                "Thunder"
              ],
              "ability": "Analytic"
            }
          ]
        }
      ]
    },
    "Meteor Falls": {
      "zone_name": "Meteor Falls",
      "zone_trainers": [
        {
          "trainer": "Old Couple John & Jay [Double]",
          "pokemon_list": [
            {
              "pokemon": "Whimsicott",
              "item": "Fairy Gem",
              "moves": [
                "Moonblast",
                "Encore",
                "Grass Whistle",
                "Tailwind"
              ],
              "ability": "Prankster"
            },
            {
              "pokemon": "Lycanroc_Midnight",
              "item": "Focus Sash",
              "moves": [
                "Stone Edge",
                "Rock Slide",
                "Close Combat",
                "Endeavor"
              ],
              "ability": "No Guard"
            },
            {
              "pokemon": "Lilligant",
              "item": "Leftovers",
              "moves": [
                "Energy Ball",
                "Hidden Power Fire",
                "Quiver Dance",
                "Sleep Powder"
              ],
              "ability": "Own Tempo"
            },
            {
              "pokemon": "Clefable",
              "item": "Iapapa Berry",
              "moves": [
                "Moonblast",
                "Fire Blast",
                "Knock Off",
                "Follow Me"
              ],
              "ability": "Unaware"
            },
            {
              "pokemon": "Heliolisk",
              "item": "Normal Gem",
              "moves": [
                "Thunder",
                "Electroweb",
                "Hyper Voice",
                "Grass Knot"
              ],
              "ability": "Dry Skin"
            },
            {
              "pokemon": "Nidoqueen",
              "item": "Life Orb",
              "moves": [
                "Sludge Wave",
                "Earth Power",
                "Blizzard",
                "Fire Blast"
              ],
              "ability": "Sheer Force"
            }
          ]
        },
        {
          "trainer": "Dragon Tamer Nicolas",
          "pokemon_list": [
            {
              "pokemon": "Gyarados",
              "item": "Wacan Berry",
              "moves": [
                "Waterfall",
                "Power Whip",
                "Earthquake",
                "Scale Shot"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Sandaconda",
              "item": "Iapapa Berry",
              "moves": [
                "Earthquake",
                "Stone Edge",
                "Glare",
                "Stealth Rock"
              ],
              "ability": "Sand Spit"
            },
            {
              "pokemon": "Metagross",
              "item": "Lum Berry",
              "moves": [
                "Meteor Mash",
                "Zen Headbutt",
                "Explosion",
                "Ice Punch"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Garchomp",
              "item": "Leftovers",
              "moves": [
                "Dual Chop",
                "Iron Tail",
                "Substitute",
                "Swords Dance"
              ],
              "ability": "Sand Veil"
            },
            {
              "pokemon": "Goodra_Hisuian",
              "item": "Assault Vest",
              "moves": [
                "Dragon Pulse",
                "Heavy Slam",
                "Fire Blast",
                "Body Press"
              ],
              "ability": "Gooey"
            },
            {
              "pokemon": "Latias",
              "item": "Latiasite",
              "moves": [
                "Psychic",
                "Mystical Fire",
                "Calm Mind",
                "Recover"
              ],
              "ability": "Levitate"
            }
          ]
        }
      ]
    },
    "Route 115": {
      "zone_name": "Route 115",
      "zone_trainers": [
        {
          "trainer": "Triathlete Kyra",
          "pokemon_list": [
            {
              "pokemon": "Togedemaru",
              "item": "Red Card",
              "moves": [
                "Zing Zap",
                "Iron Head",
                "Reversal",
                "Endeavor"
              ],
              "ability": "Sturdy"
            },
            {
              "pokemon": "Swellow",
              "item": "Choice Specs",
              "moves": [
                "Boomburst",
                "Hurricane",
                "U Turn"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Dugtrio",
              "item": "Focus Sash",
              "moves": [
                "Earthquake",
                "Night Slash",
                "Reversal",
                "Screech"
              ],
              "ability": "Arena Trap"
            },
            {
              "pokemon": "Gourgeist_Small",
              "item": "Colbur Berry",
              "moves": [
                "Power Whip",
                "Poltergeist",
                "Explosion",
                "Destiny Bond"
              ],
              "ability": "Insomnia"
            },
            {
              "pokemon": "Simipour",
              "item": "Petaya Berry",
              "moves": [
                "Hydro Pump",
                "Ice Beam",
                "Grass Knot",
                "Nasty Plot"
              ],
              "ability": "Gluttony"
            },
            {
              "pokemon": "Infernape",
              "item": "Life Orb",
              "moves": [
                "Close Combat",
                "Fire Punch",
                "Thunder Punch",
                "U Turn"
              ],
              "ability": "Iron Fist"
            }
          ]
        },
        {
          "trainer": "Battle Girl Helene",
          "pokemon_list": [
            {
              "pokemon": "Exeggutor",
              "item": "Custap Berry",
              "moves": [
                "Psychic",
                "Energy Ball",
                "Explosion",
                "Endure"
              ],
              "ability": "Harvest"
            },
            {
              "pokemon": "Hawlucha",
              "item": "Flying Gem",
              "moves": [
                "Close Combat",
                "Acrobatics",
                "Baton Pass",
                "Swords Dance"
              ],
              "ability": "Unburden"
            },
            {
              "pokemon": "Bisharp",
              "item": "Steel Gem",
              "moves": [
                "Iron Head",
                "Sucker Punch",
                "Knock Off",
                "Substitute"
              ],
              "ability": "Defiant"
            },
            {
              "pokemon": "Kleavor",
              "item": "Life Orb",
              "moves": [
                "Rock Slide",
                "Skitter Smack",
                "Agility",
                "Baton Pass"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Mamoswine",
              "item": "White Herb",
              "moves": [
                "Earthquake",
                "Icicle Crash",
                "Ice Shard",
                "Superpower"
              ],
              "ability": "Thick Fat"
            },
            {
              "pokemon": "Cinderace",
              "item": "Fire Gem",
              "moves": [
                "Pyro Ball",
                "Gunk Shot",
                "Sucker Punch",
                "Substitute"
              ],
              "ability": "Libero"
            }
          ]
        },
        {
          "trainer": "Ninja Boy Jack",
          "pokemon_list": [
            {
              "pokemon": "Wyrdeer",
              "item": "Iapapa Berry",
              "moves": [
                "Psychic",
                "Megahorn",
                "Throat Chop",
                "Thunder Wave"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Zoroark",
              "item": "Life Orb",
              "moves": [
                "Sucker Punch",
                "Knock Off",
                "Low Kick",
                "Swords Dance"
              ],
              "ability": "Illusion"
            },
            {
              "pokemon": "Ditto",
              "item": "Bright Powder",
              "moves": [
                "Transform"
              ],
              "ability": "Imposter"
            },
            {
              "pokemon": "Ferrothorn",
              "item": "Quick Claw",
              "moves": [
                "Power Whip",
                "Explosion",
                "Knock Off",
                "Curse"
              ],
              "ability": "Iron Barbs"
            },
            {
              "pokemon": "Kecleon",
              "item": "Assault Vest",
              "moves": [
                "Drain Punch",
                "Sucker Punch",
                "Knock Off",
                "Power Up Punch"
              ],
              "ability": "Color Change"
            },
            {
              "pokemon": "Wobbuffet",
              "item": "Leftovers",
              "moves": [
                "Counter",
                "Mirror Coat",
                "Destiny Bond",
                "Encore"
              ],
              "ability": "Shadow Tag"
            }
          ]
        },
        {
          "trainer": "Psychic Alix",
          "pokemon_list": [
            {
              "pokemon": "Pyroar",
              "item": "Focus Sash",
              "moves": [
                "Fire Blast",
                "Hyper Voice",
                "Endeavor",
                "Will O Wisp"
              ],
              "ability": "Unnerve"
            },
            {
              "pokemon": "Torterra",
              "item": "Yache Berry",
              "moves": [
                "Wood Hammer",
                "Earthquake",
                "Head Smash",
                "Stealth Rock"
              ],
              "ability": "Rock Head"
            },
            {
              "pokemon": "Kommo_O",
              "item": "Lum Berry",
              "moves": [
                "Dragon Claw",
                "Drain Punch",
                "Iron Tail",
                "Dragon Dance"
              ],
              "ability": "Bulletproof"
            },
            {
              "pokemon": "Sylveon",
              "item": "Lum Berry",
              "moves": [
                "Misty Explosion",
                "Hyper Voice",
                "Mystical Fire",
                "Calm Mind"
              ],
              "ability": "Pixilate"
            },
            {
              "pokemon": "Azumarill",
              "item": "Assault Vest",
              "moves": [
                "Play Rough",
                "Aqua Jet",
                "Ice Punch",
                "Knock Off"
              ],
              "ability": "Huge Power"
            },
            {
              "pokemon": "Gardevoir",
              "item": "Gardevoirite",
              "moves": [
                "Hyper Voice",
                "Aura Sphere",
                "Destiny Bond",
                "Will O Wisp"
              ],
              "ability": "Synchronize"
            }
          ]
        },
        {
          "trainer": "Black Belt Koichi",
          "pokemon_list": [
            {
              "pokemon": "Porygon2",
              "item": "Red Card",
              "moves": [
                "Tri Attack",
                "Blizzard",
                "Thunder",
                "Foul Play"
              ],
              "ability": "Analytic"
            },
            {
              "pokemon": "Yanmega",
              "item": "Choice Specs",
              "moves": [
                "Hurricane",
                "Bug Buzz",
                "U Turn"
              ],
              "ability": "Tinted Lens"
            },
            {
              "pokemon": "Ursaluna",
              "item": "Flame Orb",
              "moves": [
                "Earthquake",
                "Facade",
                "Retaliate",
                "Crunch"
              ],
              "ability": "Guts"
            },
            {
              "pokemon": "Blaziken",
              "item": "Focus Sash",
              "moves": [
                "Close Combat",
                "Blaze Kick",
                "Knock Off",
                "Swords Dance"
              ],
              "ability": "Speed Boost"
            },
            {
              "pokemon": "Conkeldurr",
              "item": "Fighting Gem",
              "moves": [
                "Mach Punch",
                "Ice Punch",
                "Thunder Punch",
                "Bulk Up"
              ],
              "ability": "Iron Fist"
            },
            {
              "pokemon": "Gallade",
              "item": "Galladite",
              "moves": [
                "Close Combat",
                "Zen Headbutt",
                "Throat Chop",
                "Hypnosis"
              ],
              "ability": "Inner Focus"
            }
          ]
        },
        {
          "trainer": "Expert Timothy",
          "pokemon_list": [
            {
              "pokemon": "Escavalier",
              "item": "Focus Sash",
              "moves": [
                "Megahorn",
                "Iron Head",
                "Drill Run",
                "Metal Burst"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Breloom",
              "item": "Fighting Gem",
              "moves": [
                "Mach Punch",
                "Bullet Seed",
                "Rock Tomb",
                "Swords Dance"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Miltank",
              "item": "Leftovers",
              "moves": [
                "Body Slam",
                "Body Press",
                "Curse",
                "Milk Drink"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Dragonite",
              "item": "Weakness Policy",
              "moves": [
                "Dual Wingbeat",
                "Earthquake",
                "Extreme Speed",
                "Dragon Dance"
              ],
              "ability": "Multiscale"
            },
            {
              "pokemon": "Greninja_Battle_Bond",
              "item": "Dark Gem",
              "moves": [
                "Hydro Pump",
                "Dark Pulse",
                "Water Shuriken",
                "Ice Beam"
              ],
              "ability": "Battle Bond"
            },
            {
              "pokemon": "Lucario",
              "item": "Lucarionite",
              "moves": [
                "Close Combat",
                "Bullet Punch",
                "Crunch",
                "Swords Dance"
              ],
              "ability": "Inner Focus"
            }
          ]
        }
      ]
    },
    "Route 110 (Trick House Door)": {
      "zone_name": "Route 110 (Trick House Door)",
      "zone_trainers": [
        {
          "trainer": "Dumbass Soupercell",
          "pokemon_list": [
            {
              "pokemon": "Dugtrio",
              "item": "Red Card",
              "moves": [
                "Memento",
                "Stealth Rock"
              ],
              "ability": "Arena Trap"
            },
            {
              "pokemon": "Pincurchin",
              "item": "Red Card",
              "moves": [
                "Memento",
                "Spikes",
                "Toxic Spikes"
              ],
              "ability": "Lightning Rod"
            },
            {
              "pokemon": "Jumpluff",
              "item": "Red Card",
              "moves": [
                "Light Screen",
                "Memento",
                "Reflect",
                "Spore"
              ],
              "ability": "Chlorophyll"
            },
            {
              "pokemon": "Shedinja",
              "item": "Sticky Barb",
              "moves": [
                "Swords Dance"
              ],
              "ability": "Wonder Guard"
            }
          ]
        }
      ]
    },
    "Reporters": {
      "zone_name": "Reporters",
      "zone_trainers": []
    },
    "split_name": "Victory Road"
  },
  "Elite Four": {
    "Pokemon League": {
      "zone_name": "Pokemon League",
      "zone_trainers": [
        {
          "trainer": "Elite Four Sidney",
          "pokemon_list": [
            {
              "pokemon": "Greninja",
              "item": "Focus Sash",
              "moves": [
                "Dark Pulse",
                "Scald",
                "Spikes",
                "Toxic Spikes"
              ],
              "ability": "Protean"
            },
            {
              "pokemon": "Necrozma",
              "item": "Leftovers",
              "moves": [
                "Photon Geyser",
                "Heat Wave",
                "Knock Off",
                "Stealth Rock"
              ],
              "ability": "Prism Armor"
            },
            {
              "pokemon": "Nidoking",
              "item": "Life Orb",
              "moves": [
                "Sludge Wave",
                "Earth Power",
                "Ice Beam",
                "Dark Pulse"
              ],
              "ability": "Sheer Force"
            },
            {
              "pokemon": "Urshifu",
              "item": "Roseli Berry",
              "moves": [
                "Close Combat",
                "Wicked Blow",
                "Sucker Punch",
                "Swords Dance"
              ],
              "ability": "Unseen Fist"
            },
            {
              "pokemon": "Yveltal",
              "item": "Dread Plate",
              "moves": [
                "Dark Pulse",
                "Oblivion Wing",
                "Heat Wave",
                "Roost"
              ],
              "ability": "Dark Aura"
            },
            {
              "pokemon": "Gyarados",
              "item": "Gyaradosite",
              "moves": [
                "Waterfall",
                "Crunch",
                "Earthquake",
                "Dragon Dance"
              ],
              "ability": "Intimidate"
            }
          ]
        },
        {
          "trainer": "Elite Four Sidney [Double]",
          "pokemon_list": [
            {
              "pokemon": "Incineroar",
              "item": "Iapapa Berry",
              "moves": [
                "Flare Blitz",
                "Knock Off",
                "Fire Spin",
                "Fake Out"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Articuno_Galarian",
              "item": "Lum Berry",
              "moves": [
                "Hurricane",
                "Freezing Glare",
                "Hypnosis",
                "Tailwind"
              ],
              "ability": "Competitive"
            },
            {
              "pokemon": "Overqwil",
              "item": "Assault Vest",
              "moves": [
                "Gunk Shot",
                "Crunch",
                "Fell Stinger",
                "Icy Wind"
              ],
              "ability": "Intimidate"
            },
            {
              "pokemon": "Urshifu",
              "item": "Focus Sash",
              "moves": [
                "Close Combat",
                "Wicked Blow",
                "Reversal",
                "Detect"
              ],
              "ability": "Unseen Fist"
            },
            {
              "pokemon": "Darkrai",
              "item": "Life Orb",
              "moves": [
                "Dark Pulse",
                "Sludge Bomb",
                "Aura Sphere",
                "Dark Void"
              ],
              "ability": "Bad Dreams"
            },
            {
              "pokemon": "Gyarados",
              "item": "Gyaradosite",
              "moves": [
                "Waterfall",
                "Power Whip",
                "Lash Out",
                "Icy Wind"
              ],
              "ability": "Intimidate"
            }
          ]
        },
        {
          "trainer": "Elite Four Phoebe",
          "pokemon_list": [
            {
              "pokemon": "Crobat",
              "item": "Flying Gem",
              "moves": [
                "Acrobatics",
                "Super Fang",
                "Hypnosis",
                "Roost"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Zoroark_Hisuian",
              "item": "Life Orb",
              "moves": [
                "Shadow Ball",
                "Flamethrower",
                "U Turn",
                "Grass Knot"
              ],
              "ability": "Illusion"
            },
            {
              "pokemon": "Gothitelle",
              "item": "Assault Vest",
              "moves": [
                "Psychic",
                "Thunderbolt",
                "Dazzling Gleam",
                "Shadow Ball"
              ],
              "ability": "Shadow Tag"
            },
            {
              "pokemon": "Lunala",
              "item": "Power Herb",
              "moves": [
                "Moongeist Beam",
                "Meteor Beam",
                "Moonblast",
                "Roost"
              ],
              "ability": "Shadow Shield"
            },
            {
              "pokemon": "Marshadow",
              "item": "Lum Berry",
              "moves": [
                "Close Combat",
                "Shadow Sneak",
                "Knock Off",
                "Bulk Up"
              ],
              "ability": "Technician"
            },
            {
              "pokemon": "Gengar",
              "item": "Gengarite",
              "moves": [
                "Sludge Wave",
                "Shadow Ball",
                "Focus Blast",
                "Destiny Bond"
              ],
              "ability": "Levitate"
            }
          ]
        },
        {
          "trainer": "Elite Four Phoebe [Double]",
          "pokemon_list": [
            {
              "pokemon": "Decidueye_Hisuian",
              "item": "Focus Sash",
              "moves": [
                "Close Combat",
                "Leaf Blade",
                "Knock Off",
                "Tailwind"
              ],
              "ability": "Scrappy"
            },
            {
              "pokemon": "Chandelure",
              "item": "Kasib Berry",
              "moves": [
                "Fire Blast",
                "Heat Wave",
                "Shadow Ball",
                "Protect"
              ],
              "ability": "Shadow Tag"
            },
            {
              "pokemon": "Blastoise",
              "item": "Iapapa Berry",
              "moves": [
                "Muddy Water",
                "Focus Blast",
                "Follow Me",
                "Fake Out"
              ],
              "ability": "Torrent"
            },
            {
              "pokemon": "Lunala",
              "item": "Power Herb",
              "moves": [
                "Moongeist Beam",
                "Meteor Beam",
                "Dazzling Gleam",
                "Tailwind"
              ],
              "ability": "Shadow Shield"
            },
            {
              "pokemon": "Giratina_Origin",
              "item": "Griseous Orb",
              "moves": [
                "Poltergeist",
                "Dragon Pulse",
                "Shadow Sneak",
                "Flamethrower"
              ],
              "ability": "Levitate"
            },
            {
              "pokemon": "Gengar",
              "item": "Gengarite",
              "moves": [
                "Sludge Wave",
                "Shadow Ball",
                "Focus Blast",
                "Protect"
              ],
              "ability": "Levitate"
            }
          ]
        },
        {
          "trainer": "Elite Four Glacia",
          "pokemon_list": [
            {
              "pokemon": "Mamoswine",
              "item": "Focus Sash",
              "moves": [
                "Earthquake",
                "Icicle Crash",
                "Stone Edge",
                "Stealth Rock"
              ],
              "ability": "Oblivious"
            },
            {
              "pokemon": "Abomasnow",
              "item": "Abomasite",
              "moves": [
                "Blizzard",
                "Giga Drain",
                "Focus Blast",
                "Aurora Veil"
              ],
              "ability": "Soundproof"
            },
            {
              "pokemon": "Arctovish",
              "item": "Ice Gem",
              "moves": [
                "Fishious Rend",
                "Blizzard",
                "Icicle Crash",
                "Whirlpool"
              ],
              "ability": "Slush Rush"
            },
            {
              "pokemon": "Enamorus_Therian",
              "item": "Leftovers",
              "moves": [
                "Moonblast",
                "Mystical Fire",
                "Calm Mind",
                "Rest"
              ],
              "ability": "Overcoat"
            },
            {
              "pokemon": "Kyurem_White",
              "item": "Assault Vest",
              "moves": [
                "Dragon Pulse",
                "Ice Beam",
                "Fusion Flare",
                "Earth Power"
              ],
              "ability": "Turboblaze"
            },
            {
              "pokemon": "Calyrex_Ice_Rider",
              "item": "Custap Berry",
              "moves": [
                "Glacial Lance",
                "High Horsepower",
                "Substitute",
                "Swords Dance"
              ],
              "ability": "As One (Glastrier)"
            }
          ]
        },
        {
          "trainer": "Elite Four Glacia [Double]",
          "pokemon_list": [
            {
              "pokemon": "Magearna",
              "item": "Occa Berry",
              "moves": [
                "Flash Cannon",
                "Dazzling Gleam",
                "Encore",
                "Trick Room"
              ],
              "ability": "Soul-Heart"
            },
            {
              "pokemon": "Crabominable",
              "item": "Focus Sash",
              "moves": [
                "Close Combat",
                "Ice Hammer",
                "Reversal",
                "Crabhammer"
              ],
              "ability": "Hyper Cutter"
            },
            {
              "pokemon": "Abomasnow",
              "item": "Abomasite",
              "moves": [
                "Blizzard",
                "Giga Drain",
                "Earth Power",
                "Aurora Veil"
              ],
              "ability": "Snow Warning"
            },
            {
              "pokemon": "Porygon2",
              "item": "Eviolite",
              "moves": [
                "Tri Attack",
                "Blizzard",
                "Psychic",
                "Trick Room"
              ],
              "ability": "Download"
            },
            {
              "pokemon": "Calyrex_Ice_Rider",
              "item": "Ice Gem",
              "moves": [
                "Glacial Lance",
                "Zen Headbutt",
                "Close Combat",
                "Trick Room"
              ],
              "ability": "As One (Glastrier)"
            },
            {
              "pokemon": "Kyurem_Black",
              "item": "Room Service",
              "moves": [
                "Draco Meteor",
                "Icicle Spear",
                "Fusion Bolt",
                "Stone Edge"
              ],
              "ability": "Teravolt"
            }
          ]
        },
        {
          "trainer": "Elite Four Drake",
          "pokemon_list": [
            {
              "pokemon": "Dragapult",
              "item": "Life Orb",
              "moves": [
                "Dragon Darts",
                "Shadow Ball",
                "Fire Blast",
                "Steel Wing"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Crawdaunt",
              "item": "Focus Sash",
              "moves": [
                "Crabhammer",
                "Lash Out",
                "Close Combat",
                "Dragon Dance"
              ],
              "ability": "Adaptability"
            },
            {
              "pokemon": "Zygarde_50_Power_Construct",
              "item": "Haban Berry",
              "moves": [
                "Thousand Arrows",
                "Dragon Claw",
                "Coil",
                "Glare"
              ],
              "ability": "Power Construct"
            },
            {
              "pokemon": "Suicune",
              "item": "Leftovers",
              "moves": [
                "Scald",
                "Ice Beam",
                "Calm Mind",
                "Substitute"
              ],
              "ability": "Pressure"
            },
            {
              "pokemon": "Reshiram",
              "item": "Lum Berry",
              "moves": [
                "Blue Flare",
                "Dragon Pulse",
                "Earth Power",
                "Roost"
              ],
              "ability": "Turboblaze"
            },
            {
              "pokemon": "Salamence",
              "item": "Salamencite",
              "moves": [
                "Return",
                "Earthquake",
                "Dragon Dance",
                "Roost"
              ],
              "ability": "Intimidate"
            }
          ]
        },
        {
          "trainer": "Elite Four Drake [Double]",
          "pokemon_list": [
            {
              "pokemon": "Dragapult",
              "item": "Focus Sash",
              "moves": [
                "Dragon Darts",
                "Shadow Ball",
                "Light Screen",
                "Reflect"
              ],
              "ability": "Clear Body"
            },
            {
              "pokemon": "Suicune",
              "item": "Leftovers",
              "moves": [
                "Scald",
                "Snarl",
                "Calm Mind",
                "Tailwind"
              ],
              "ability": "Inner Focus"
            },
            {
              "pokemon": "Zygarde_50_Power_Construct",
              "item": "Leftovers",
              "moves": [
                "Thousand Waves",
                "Breaking Swipe",
                "Extreme Speed",
                "Coil"
              ],
              "ability": "Power Construct"
            },
            {
              "pokemon": "Charizard",
              "item": "Life Orb",
              "moves": [
                "Heat Wave",
                "Hurricane",
                "Aura Sphere",
                "Tailwind"
              ],
              "ability": "Blaze"
            },
            {
              "pokemon": "Zekrom",
              "item": "White Herb",
              "moves": [
                "Draco Meteor",
                "Bolt Strike",
                "Scale Shot",
                "Protect"
              ],
              "ability": "Teravolt"
            },
            {
              "pokemon": "Salamence",
              "item": "Salamencite",
              "moves": [
                "Dragon Pulse",
                "Hyper Voice",
                "Flamethrower",
                "Tailwind"
              ],
              "ability": "Intimidate"
            }
          ]
        },
        {
          "trainer": "Champion Wallace",
          "pokemon_list": [
            {
              "pokemon": "Kyogre",
              "item": "Blue Orb",
              "moves": [
                "Origin Pulse",
                "Liquidation",
                "Thunder",
                "Ice Beam"
              ],
              "ability": "Drizzle"
            },
            {
              "pokemon": "Barraskewda",
              "item": "Choice Band",
              "moves": [
                "Liquidation",
                "Flip Turn"
              ],
              "ability": "Swift Swim"
            },
            {
              "pokemon": "Goodra_Hisuian",
              "item": "Leftovers",
              "moves": [
                "Heavy Slam",
                "Aqua Tail",
                "Curse",
                "Rest"
              ],
              "ability": "Shell Armor"
            },
            {
              "pokemon": "Palkia",
              "item": "Scope Lens",
              "moves": [
                "Draco Meteor",
                "Hydro Pump",
                "Earth Power",
                "Focus Energy"
              ],
              "ability": "Pressure"
            },
            {
              "pokemon": "Manaphy",
              "item": "Leftovers",
              "moves": [
                "Scald",
                "Energy Ball",
                "Ice Beam",
                "Tail Glow"
              ],
              "ability": "Hydration"
            },
            {
              "pokemon": "Swampert",
              "item": "Swampertite",
              "moves": [
                "Earthquake",
                "Liquidation",
                "Stone Edge",
                "Ice Punch"
              ],
              "ability": "Torrent"
            }
          ]
        }
      ]
    },
    "split_name": "Elite Four"
  }
}
# brawly_zone = ["Route 102", "Route 104 (South)", "Route 106", "Route 109", "Route 110 (South)",
#                "Slateport Museum", "Dewford Gym"]

# roxanne_zone = ["Petalburg Woods", "Route 104 (North)", "Route 116", "Rustboro Gym"]

# wattson_zone = ["Rustuf Tunnel", "Route 117", "Route 111 (South)", "Mauville's Gym"]

# norman_zone = ["Route 110 (North)", "Route 110 (Cycling Road)", "Route 103 (East)", "Petalburg Gym"]

# flannery_zone = ["Route 111 (Desert)", "Route 111 (North)", "Route 113", "Fallarbor", "Route 114", 
#                  "Route 115", "Route 112 (North)", "Route 112 (South)", "Mt. Chimney", "Jagged Pass",
#                  "Lavaridge Gym"]
# winona_zone = ["Seashore House", "Route 105 (Optionals)", "Route 106 (Optionals)", 
#                "Route 107 (Optionals)", "Route 108 (Optionals)", "Route 109 (Optionals)",
#                "Route 118", "Route 119 (West)", "Weather Institute", "Route 119 (East)", 
#                "Route 120 (North)", "Fortree Gym"]

# tnl_zone = ["Route 120 (South)", "Route 121", "Lilycove", "Mt. Pyre", "Magma Hideout",
#             "Aqua Hideout", "Route 124", "Route 125 (Optionals)", "Mossdeep Gym" ]

# juan_zone = ["Mossdeep Space Center", "Route 124 (South)", "Route 126", 
#              "Route 126 (Optionals)", "Route 127", "Route 127 (Optionals)", "Route 128",
#             "Seafloor Cavern", "Route 129", "Route 130",
#              "Route 131",  "Route 133", "Sootopolis Gym"]

# vr_split = ["Victory Road", "Route 123", "Meteor Falls", "Route 115"]

# e4_split = ["Pokemon League"]

split_list = ["Brawly", "Roxanne", "Wattson",
              "Norman", "Flannery",
              "Winona", "TnL",
              "Juan", "Victory Road",
              "Elite Four"]

# file_path = r"Split_json\Full\BattleData.json"

                                



# def load_data():
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             data = json.load(file)

#     except FileNotFoundError:
#         print(f"Error: The file '{file_path}' was not found.")
#     except json.JSONDecodeError:
#         print(f"Error: Could not decode JSON from the file '{file_path}'. Check if the file has a valid JSON format.")


def is_valid_split_name(split_name):
    if (split_name in split_list):
        return True
    else:
        print(f"{split_name} is not a valid split name")
        return False

def is_valid_zone_for_split(split_name, zone_name):
    if(is_valid_split_name(split_name=split_name)):
        if (zone_name in get_zones_in_split(split_name=split_name)):
            return True
        else:
            print(f"{zone_name} is not a valid zone name for {split_name} Split")
            return False
    else:
        print("{split_name} or {zone_name} is invalid")
        return False


def get_splits():
    return find_key(data, "split_name")

def get_zones_in_split(split_name):
    if(is_valid_split_name(split_name=split_name)):
        return find_key(data[split_name], "zone_name")
    return False

def get_trainers_in_split(split_name):
    if not is_valid_split_name(split_name=split_name):
        return False
    split = data[split_name]
    return find_key(split, "trainer")


def get_trainers_in_split_in_zone(split_name, zone_name):
    if not is_valid_zone_for_split(split_name=split_name, zone_name=zone_name):
        return False
    split_data = data[split_name]
    zone_data = split_data[zone_name]
    return find_key(zone_data, "trainer")

def get_pokemon_from_trainers_in_split_in_zone(split_name, zone_name):
    if not is_valid_zone_for_split(split_name=split_name, zone_name=zone_name):
        return False
    return find_key_chain(data, [split_name, zone_name, "zone_trainers", "pokemon_list", "pokemon"])
    
def get_trainer_dictionary(trainer_name):
    return find_key_value(data, "trainer", trainer_name)

def get_pokemon_from_trainer_name(trainer_name):
    trainer_data = get_trainer_dictionary(trainer_name=trainer_name)
    #return find_value(trainer_data, "pokemon_list")
    return find_key(trainer_data, 'pokemon')

def get_pokemon_moves_from_trainer_name(trainer_name):
    trainer_data = get_trainer_dictionary(trainer_name=trainer_name)
    return find_key(trainer_data, 'moves')

def get_pokemon_items_from_trainer_name(trainer_name):
    trainer_data = get_trainer_dictionary(trainer_name=trainer_name)
    return find_key(trainer_data, 'item')

def get_pokemon_ability_from_trainer_name(trainer_name):
    trainer_data = get_trainer_dictionary(trainer_name=trainer_name)
    return find_key(trainer_data, 'ability')

if __name__ == "__main__":
    #print(find_key(data["Brawly"], "zone_name"))
    #trainer_list = find_key(data, "trainer")
    # brawly_split = data["Brawly"]

    # brawly_zone = brawly_split["Dewford Gym"]
    # pprint.pprint(brawly_zone)
    # brawly_trainer = find_key(brawly_split, "trainer")
    # pprint.pprint(brawly_trainer)

    # brawly_zones = find_key(brawly_split, "zone_name")
    # pprint.pprint(brawly_zones)
    print(is_valid_zone_for_split("Brawly", "Route 1023"))
    pass