"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

OUT_DOMAIN_TRAIN_SLOT_VALUE_DICT = {
"hotel-internet": ['yes'] ,
"hotel-type": ['hotel', 'guesthouse'],
"hotel-parking": ['yes'] ,
"hotel-pricerange": ['moderate', 'cheap', 'expensive'] ,
"hotel-book day": ["march 11th", "march 12th", "march 13th", "march 14th", "march 15th", "march 16th", "march 17th", 
                   "march 18th", "march 19th", "march 20th"],
"hotel-book people": ["20","21","22","23","24","25","26","27","28","29"],
"hotel-book stay": ["20","21","22","23","24","25","26","27","28","29"],
"hotel-area": ['south', 'north', 'west', 'east', 'centre'],
"hotel-stars": ['0', '1', '2', '3', '4', '5'] ,
"hotel-name":["moody moon", "four seasons hotel", "knights inn", "travelodge", "jack summer inn", "paradise point resort"],
"restaurant-area": ['south', 'north', 'west', 'east', 'centre'],
"restaurant-food": ['asian fusion', 'burger', 'pasta', 'ramen', 'taiwanese'],
"restaurant-pricerange": ['moderate', 'cheap', 'expensive'] ,
"restaurant-name": ["buddha bowls","pizza my heart","pho bistro","sushiya express","rockfire grill","itsuki restaurant"],
"restaurant-book day": ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"],
"restaurant-book people": ["20","21","22","23","24","25","26","27","28","29"],
"restaurant-book time":["19:01","18:06","17:11","19:16","18:21","17:26","19:31","18:36","17:41","19:46","18:51","17:56",
                        "7:00 pm","6:07 pm","5:12 pm","7:17 pm","6:17 pm","5:27 pm","7:32 pm","6:37 pm","5:42 pm","7:47 pm","6:52 pm","5:57 pm",
                        "11:00 am","11:05 am","11:10 am","11:15 am","11:20 am","11:25 am","11:30 am","11:35 am","11:40 am","11:45 am","11:50 am",
                        "11:55 am"],
"taxi-arriveby": [ "17:26","19:31","18:36","17:41","19:46","18:51","17:56",
                    "7:00 pm","6:07 pm","5:12 pm","7:17 pm","6:17 pm","5:27 pm",
                    "11:30 am","11:35 am","11:40 am","11:45 am","11:50 am","11:55 am"],
"taxi-leaveat":  [ "19:01","18:06","17:11","19:16","18:21",
                    "7:32 pm","6:37 pm","5:42 pm","7:47 pm","6:52 pm","5:57 pm",
                    "11:00 am","11:05 am","11:10 am","11:15 am","11:20 am","11:25 am"],
"taxi-departure":   ["moody moon", "four seasons hotel", "knights inn", "travelodge", "jack summer inn", "paradise point resort"],
"taxi-destination": ["buddha bowls","pizza my heart","pho bistro","sushiya express","rockfire grill","itsuki restaurant"],
"train-arriveby": [ "17:26","19:31","18:36","17:41","19:46","18:51","17:56",
                    "7:00 pm","6:07 pm","5:12 pm","7:17 pm","6:17 pm","5:27 pm",
                    "11:30 am","11:35 am","11:40 am","11:45 am","11:50 am","11:55 am"],
"train-leaveat":   [ "19:01","18:06","17:11","19:16","18:21",
                    "7:32 pm","6:37 pm","5:42 pm","7:47 pm","6:52 pm","5:57 pm",
                    "11:00 am","11:05 am","11:10 am","11:15 am","11:20 am","11:25 am"],
"train-departure": ["gilroy","san martin","morgan hill","blossom hill","college park","santa clara","lawrence","sunnyvale"],
"train-destination":["mountain view","san antonio","palo alto","menlo park","hayward park","san mateo","broadway","san bruno"],
"train-day":       ["march 11th", "march 12th", "march 13th", "march 14th", "march 15th", "march 16th", "march 17th", 
                   "march 18th", "march 19th", "march 20th"],

"train-book people":["20","21","22","23","24","25","26","27","28","29"],
"attraction-area": ['south', 'north', 'west', 'east', 'centre'],
"attraction-name": ["grand canyon","golden gate bridge","niagara falls","kennedy space center","pike place market","las vegas strip"],
"attraction-type": ['historical landmark', 'aquaria', 'beach', 'castle','art gallery']
}

IN_DOMAIN_TEST_SLOT_VALUE_DICT = {
"hotel-internet": ['yes'],
"hotel-type": ['hotel', 'guesthouse'] , 
"hotel-parking": ['yes'] ,
"hotel-pricerange": ['moderate', 'cheap', 'expensive'] , 
"hotel-book day": ['friday', 'tuesday', 'thursday', 'saturday', 'monday', 'sunday', 'wednesday'] ,
"hotel-book people": ['1', '2', '3', '4','5', '6', '7','8'],
"hotel-book stay": ['1', '2', '3', '4','5', '6', '7','8'] ,
"hotel-name": ['alpha milton', 'flinches bed and breakfast', 'express holiday inn by cambridge', 
                'wankworth house', 'alexander b and b', 'the gonville hotel'],
"hotel-stars": ['0', '1', '3', '2', '4', '5'] ,
"hotel-area": ['south', 'east', 'west', 'north', 'centre'] ,
"restaurant-area": ['south', 'east', 'west', 'north', 'centre'] ,
"restaurant-food": ['europeon', 'brazliian', 'weish'] ,
"restaurant-pricerange": ['moderate', 'cheap', 'expensive'] ,
"restaurant-name":['pizza hut in cherry', 'the nirala', 'barbakan', 'the golden house', 'michaelhouse', 'bridge', 'varsity restaurant', 
                    'loch', 'the peking', 'charlie', 'cambridge lodge', 'maharajah tandoori'] ,
"restaurant-book day": ['friday', 'tuesday', 'thursday', 'saturday', 'monday', 'sunday', 'wednesday'] ,
"restaurant-book people":['8', '6', '7', '1', '3', '2', '4', '5'] ,
"restaurant-book time": ['14:40', '19:00', '15:15', '9:30', '7 pm', '11 am', '8:45'] ,
"taxi-arriveby": ['08:30', '9:45'] ,
"taxi-leaveat":['7 pm', '3:00'],
"taxi-departure": ['aylesbray lodge', 'fitzbillies', 'uno', 'zizzi cambridge', 'express by holiday inn', 'great saint marys church',  'county folk museum', 
        'riverboat', 'bishops stortford', 'caffee uno', 'hong house', 'gandhi', 'cambridge arts', 'the hotpot', 'regency gallery', 'saint johns chop shop house'] ,
"taxi-destination": ['ashley', 'all saints', "de luca cucina and bar's", 'the lensfield hotel', 'oak bistro', 'broxbourne', 'sleeperz hotel', "saint catherine's college"],
"train-arriveby":['4:45 pm', '18:35', '21:08', '19:54', '10:08', '13:06', '15:24', '07:08', '16:23', '8:56', '09:01', '10:23', '10:00 am', '16:44', '6:15', '06:01', '8:54',
                '21:51', '16:07', '12:43', '20:08', '08:23', '12:56', '17:23', '11:32', '20:54', '20:06', '14:24', '18:10', '20:38', '16:06', '3:00', '22:06', '20:20', '17:51', 
                '19:52', '7:52', '07:44', '16:08'],
"train-leaveat": ['13:36', '15:17', '14:21', '3:15 pm', '6:10 am', '14:40', '5:40', '13:40', '17:11', '13:50', '5:11', '11:17', '5:01', '13:24', '5:35', '07:00', '8:08', '7:40', '11:54', 
                '12:06', '07:01', '18:09', '13:17', '21:45', '06:40', '01:44', '9:17', '20:21', '20:40', '08:11', '07:35', '14:19', '1 pm', '19:17', '19:48', '19:50', '10:36', '09:19',
                 '19:35', '8:06', '05:29', '17:50', '15:16', '09:17', '7:35', '5:29', '17:16', '14:01', '10:21', '05:01', '15:39', '15:01', '10:11', '08:01'],        
"train-departure": ['london liverpool street', 'kings lynn', 'norwich', 'birmingham new street', 'london kings cross','broxbourne'],
"train-destination":['bishops stortford', 'cambridge', 'ely', 'stansted airport', 'peterborough', 'leicester', 'stevenage'],
"train-day": ['friday', 'tuesday', 'thursday', 'monday', 'saturday', 'sunday', 'wednesday'],
"train-book people": ['9'],
"attraction-name": ['the cambridge arts theatre', 'the churchill college', 'the castle galleries', 'cambridge', "saint catherine's college", 'street', 'corn cambridge exchange', 
                    'fitzwilliam', 'cafe jello museum'],
"attraction-area": ['south', 'east', 'west', 'north', 'centre'],
"attraction-type": ['concerthall', 'museum', 'entertainment', 'college', 'multiple sports', 'hiking', 'architecture', 'theatre', 'cinema', 'swimmingpool', 'boat', 'nightclub', 'park']
}

OUT_DOMAIN_TEST_SLOT_VALUE_DICT = {
    "hotel-internet": ['yes'] ,
    "hotel-type": ['hotel', 'guesthouse'], 
    "hotel-parking": ['yes'] ,
    "hotel-pricerange": ['moderate', 'cheap', 'expensive'] ,
    "hotel-book day": ["april 11th", "april 12th", "april 13th", "april 14th", "april 15th", "april 16th", "april 17th", 
                       "april 18th", "april 19th", "april 20th"],
    "hotel-book people": ["30","31","32","33","34","35","36","37","38","39"],
    "hotel-book stay": ["30","31","32","33","34","35","36","37","38","39"],
    "hotel-area": ['south', 'east', 'west', 'north', 'centre'],
    "hotel-stars": ['0', '1', '2', '3', '4', '5'] ,
    "hotel-name":["white rock hotel", "jade bay resort", "grand hyatt", "hilton garden inn","cottage motel","mandarin oriental"],
    "restaurant-area": ['south', 'east', 'west', 'north', 'centre'],
    "restaurant-food": ["sichuan", "fish", "noodle", "lobster", "burrito", "dumpling", "curry","taco"],
    "restaurant-pricerange": ['moderate', 'cheap', 'expensive'] ,
    "restaurant-name": ["lure fish house","black sheep restaurant","palapa restaurant", "nikka ramen","sun sushi","super cucas"],
    "restaurant-book day": ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"],
    "restaurant-book people": ["30","31","32","33","34","35","36","37","38","39"],
    "restaurant-book time":["20:02","21:07","22:12","20:17","21:22","22:27","20:32","21:37","22:42","20:47","21:52","22:57",
                            "8:00 pm","9:04 pm","10:09 pm","8:14 pm","9:19 pm","10:24 pm","8:29 pm","9:34 pm","10:39 pm","8:44 pm","9:49 pm","10:54 pm",
                            "10:00 am","10:06 am","10:11 am","10:16 am","10:21 am","10:26 am","10:31 am","10:36 am","10:41 am","10:46 am","10:51 am","10:56 am"],
    "taxi-arriveby":    ["20:02","21:07","22:12","20:17","21:22","22:27",
                         "9:34 pm","10:39 pm","8:44 pm","9:49 pm","10:54 pm",
                        "10:00 am","10:06 am","10:11 am","10:16 am","10:21 am","10:26 am"],
    "taxi-leaveat":     ["21:37","22:42","20:47","21:52","22:57",
                        "8:00 pm","9:04 pm","10:09 pm","8:14 pm","9:19 pm","10:24 pm","8:29 pm",
                        "10:31 am","10:36 am","10:41 am","10:46 am","10:51 am","10:56 am"],
    "taxi-departure":   ["lure fish house","black sheep restaurant","palapa restaurant", "nikka ramen","sun sushi","super cucas"],
    "taxi-destination": ["white rock hotel", "jade bay resort", "grand hyatt", "hilton garden inn","cottage motel","mandarin oriental"],
    "train-departure": ["northridge","camarillo","oxnard","morepark","simi valley","chatsworth","van nuys","glendale"],
    "train-destination":["norwalk","buena park","fullerton","santa ana","tustin","irvine","san clemente","oceanside"],
    "train-arriveby": ["20:02","21:07","22:12","20:17","21:22","22:27",
                         "9:34 pm","10:39 pm","8:44 pm","9:49 pm","10:54 pm",
                        "10:00 am","10:06 am","10:11 am","10:16 am","10:21 am","10:26 am"],
    "train-day":        ["april 11th", "april 12th", "april 13th", "april 14th", "april 15th", "april 16th", "april 17th", 
                       "april 18th", "april 19th", "april 20th"],

    "train-leaveat":   ["21:37","22:42","20:47","21:52","22:57",
                        "8:00 pm","9:04 pm","10:09 pm","8:14 pm","9:19 pm","10:24 pm","8:29 pm",
                        "10:31 am","10:36 am","10:41 am","10:46 am","10:51 am","10:56 am"],

    "train-book people":["30","31","32","33","34","35","36","37","38","39"],
    "attraction-area": ['south', 'east', 'west', 'north', 'centre'],
    "attraction-name": ["statue of liberty","empire state building","mount rushmore","brooklyn bridge","lincoln memorial","times square"],
    "attraction-type": ["temple", "zoo", "library", "skyscraper","monument"]
}


SLOT_VALUE_DICT = {
    "out_domain_train" : OUT_DOMAIN_TRAIN_SLOT_VALUE_DICT,
    "out_domain_test" : OUT_DOMAIN_TEST_SLOT_VALUE_DICT,
    "in_domain_test":IN_DOMAIN_TEST_SLOT_VALUE_DICT,
}


FREQ_SLOT_COMBINE_DICT = {
    "hotel-internet": ["hotel-area","hotel-parking","hotel-pricerange","hotel-stars","hotel-type"] ,
    "hotel-type": ["hotel-area","hotel-internet","hotel-parking","hotel-pricerange","hotel-stars"] , 
    "hotel-parking": ["hotel-area","hotel-internet","hotel-pricerange","hotel-stars","hotel-type"] ,
    "hotel-pricerange": ["hotel-area","hotel-internet","hotel-parking","hotel-stars","hotel-type"] , 
    "hotel-book day": ["hotel-book people","hotel-book stay"] ,
    "hotel-book people": ["hotel-book day","hotel-book stay"] ,
    "hotel-book stay": ["hotel-book day","hotel-book people"] ,
    "hotel-stars": ["hotel-area","hotel-internet","hotel-parking","hotel-pricerange","hotel-type"] ,
    "hotel-area": ["hotel-internet","hotel-parking","hotel-pricerange","hotel-stars","hotel-type"] ,
    "hotel-name": ["hotel-book day","hotel-book people","hotel-book stay"],
    "restaurant-area": ["restaurant-food","restaurant-pricerange"] ,
    "restaurant-food": ["restaurant-area","restaurant-pricerange"] ,
    "restaurant-pricerange": ["restaurant-area","restaurant-food"] ,
    "restaurant-name":["restaurant-book day","restaurant-book people","restaurant-book time"] ,
    "restaurant-book day": ["restaurant-book people","restaurant-book time"] ,
    "restaurant-book people":["restaurant-book day","restaurant-book time"] ,
    "restaurant-book time": ["restaurant-book day","restaurant-book people"] ,
    "taxi-arriveby": ["taxi-leaveat","train-book people"] ,
    "taxi-leaveat":["taxi-arriveby","train-book people"],
    "taxi-departure": ["taxi-destination","taxi-leaveat","taxi-arriveby"] ,
    "taxi-destination": ["taxi-departure","taxi-arriveby","taxi-leaveat"],
    "train-arriveby":["train-day","train-leaveat","train-book people"],
    "train-departure": ["train-arriveby","train-leaveat","train-destination","train-day","train-book people"],
    "train-destination":["train-arriveby","train-leaveat","train-departure","train-day","train-book people"],
    "train-day": ["train-arriveby","train-leaveat","train-book people"],
    "train-leaveat": ["train-day"],
    "train-book people": [],
    "attraction-name": [],
    "attraction-area": ["attraction-type"],
    "attraction-type": ["attraction-area"]
    }

NEU_SLOT_COMBINE_DICT = {'hotel-internet': ['hotel-book day','hotel-name','hotel-book stay','hotel-pricerange','hotel-stars','hotel-area','hotel-book people','hotel-type','hotel-parking'],
 'hotel-area': ['hotel-book day','hotel-name','hotel-book stay','hotel-pricerange','hotel-stars','hotel-book people','hotel-internet','hotel-type','hotel-parking'],
 'hotel-parking': ['hotel-book day','hotel-name','hotel-book stay','hotel-pricerange','hotel-stars','hotel-area','hotel-book people','hotel-internet','hotel-type'],
 'hotel-pricerange': ['hotel-book day','hotel-name','hotel-book stay','hotel-stars','hotel-area','hotel-book people','hotel-internet','hotel-type','hotel-parking'],
 'hotel-stars': ['hotel-book day','hotel-name','hotel-book stay','hotel-pricerange','hotel-area','hotel-book people','hotel-internet','hotel-type','hotel-parking'],
 'hotel-type': ['hotel-book day','hotel-book stay','hotel-pricerange','hotel-stars','hotel-area','hotel-book people','hotel-internet','hotel-parking'],
 'hotel-name': ['hotel-book day','hotel-book stay','hotel-pricerange','hotel-stars','hotel-area','hotel-book people','hotel-internet','hotel-parking'],
 'hotel-book day': ['hotel-name','hotel-book stay','hotel-pricerange','hotel-stars','hotel-area','hotel-book people','hotel-internet','hotel-type','hotel-parking'],
 'hotel-book people': ['hotel-book day','hotel-name','hotel-book stay','hotel-pricerange','hotel-stars','hotel-area','hotel-internet','hotel-type','hotel-parking'],
 'hotel-book stay': ['hotel-book day','hotel-name','hotel-pricerange','hotel-stars','hotel-area','hotel-book people','hotel-internet','hotel-type','hotel-parking'],
 'restaurant-area': ['restaurant-book day','restaurant-name','restaurant-food','restaurant-book people','restaurant-book time','restaurant-pricerange'],
 'restaurant-food': ['restaurant-book day','restaurant-book people','restaurant-book time','restaurant-area','restaurant-pricerange'],
 'restaurant-pricerange': ['restaurant-book day','restaurant-name','restaurant-food','restaurant-book people','restaurant-book time','restaurant-area'],
 'restaurant-name': ['restaurant-book day','restaurant-book people','restaurant-book time','restaurant-area','restaurant-pricerange'],
 'restaurant-book day': ['restaurant-name','restaurant-food','restaurant-book people','restaurant-book time','restaurant-area','restaurant-pricerange'],
 'restaurant-book people': ['restaurant-book day','restaurant-name','restaurant-food','restaurant-book time','restaurant-area','restaurant-pricerange'],
 'restaurant-book time': ['restaurant-book day','restaurant-name','restaurant-food','restaurant-book people','restaurant-area','restaurant-pricerange'],
 'taxi-departure': ['taxi-destination', 'taxi-leaveat', 'taxi-arriveby'],
 'taxi-destination': ['taxi-departure', 'taxi-leaveat', 'taxi-arriveby'],
 'taxi-leaveat': ['taxi-departure', 'taxi-destination', 'taxi-arriveby'],
 'taxi-arriveby': ['taxi-departure', 'taxi-destination', 'taxi-leaveat'],
 'train-arriveby': ['train-book people','train-day','train-leaveat','train-departure','train-destination'],
 'train-leaveat': ['train-book people','train-arriveby','train-day','train-departure','train-destination'],
 'train-departure': ['train-book people','train-arriveby','train-day','train-leaveat','train-destination'],
 'train-destination': ['train-book people','train-arriveby','train-day','train-leaveat','train-departure'],
 'train-day': ['train-book people','train-arriveby','train-leaveat','train-departure','train-destination'],
 'train-book people': ['train-arriveby','train-day','train-leaveat','train-departure','train-destination'],
 'attraction-name': ['attraction-area'],
 'attraction-area': ['attraction-type'],
 'attraction-type': ['attraction-area']}


RARE_SLOT_COMBINE_DICT = {'hotel-internet': ['hotel-book people','hotel-book day','hotel-name','hotel-book stay'],
 'hotel-area': ['hotel-book people','hotel-book day','hotel-name','hotel-book stay'],
 'hotel-parking': ['hotel-book people','hotel-book day','hotel-name','hotel-book stay'],
 'hotel-pricerange': ['hotel-book people','hotel-book day','hotel-name','hotel-book stay'],
 'hotel-stars': ['hotel-book people','hotel-book day','hotel-name','hotel-book stay'],
 'hotel-type': ['hotel-book people','hotel-book day','hotel-book stay'],
 'hotel-name': ['hotel-pricerange','hotel-stars','hotel-area','hotel-internet','hotel-parking'],
 'hotel-book day': ['hotel-name','hotel-pricerange','hotel-stars','hotel-area','hotel-internet','hotel-type','hotel-parking'],
 'hotel-book people': ['hotel-name','hotel-pricerange','hotel-stars','hotel-area','hotel-internet','hotel-type','hotel-parking'],
 'hotel-book stay': ['hotel-name','hotel-pricerange','hotel-stars','hotel-area','hotel-internet','hotel-type','hotel-parking'],
 'restaurant-area': ['restaurant-book day','restaurant-name','restaurant-book time','restaurant-book people'],
 'restaurant-food': ['restaurant-book day','restaurant-book time','restaurant-book people'],
 'restaurant-pricerange': ['restaurant-book day','restaurant-name','restaurant-book time','restaurant-book people'],
 'restaurant-name': ['restaurant-area','restaurant-pricerange'],
 'restaurant-book day': ['restaurant-name','restaurant-area','restaurant-food','restaurant-pricerange'],
 'restaurant-book people': ['restaurant-name','restaurant-area','restaurant-food','restaurant-pricerange'],
 'restaurant-book time': ['restaurant-name','restaurant-area','restaurant-food','restaurant-pricerange'],
 'taxi-departure': [],
 'taxi-destination': [],
 'taxi-leaveat': ['taxi-departure', 'taxi-destination'],
 'taxi-arriveby': ['taxi-departure', 'taxi-destination'],
 'train-arriveby': ['train-destination', 'train-departure'],
 'train-leaveat': ['train-destination','train-book people','train-arriveby','train-departure'],
 'train-departure': [],
 'train-destination': [],
 'train-day': ['train-destination', 'train-departure'],
 'train-book people': ['train-arriveby','train-departure','train-destination','train-day','train-leaveat'],
 'attraction-name': ['attraction-area'],
 'attraction-area': ['attraction-name'],
 'attraction-type': []}

SLOT_COMBINE_DICT = {
    "freq" : FREQ_SLOT_COMBINE_DICT,
    "neu":NEU_SLOT_COMBINE_DICT,
    "rare" : RARE_SLOT_COMBINE_DICT,
}
