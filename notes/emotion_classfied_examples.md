# Emotion Detection on Truths

I used the model_name = "j-hartmann/emotion-english-distilroberta-base"@misc{hartmann2022emotionenglish,
  author={Hartmann, Jochen},
  title={Emotion English DistilRoBERTa-base},
  year={2022},
  howpublished = {\url{https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/}},
} to classify emotions on the Truth Sozial dataset.
The model was trained on 6 emotion data sets and provides following labels:
![alt text](image.png)



this are random classifikation examples for each label, 
from the first 2000 post in the truths_cleand.tsv Data.


📊 Histogramm-Analyse

Neutral dominiert stark, dicht gefolgt von Anger.

Disgust, Sadness, Surprise sehr niedrig.

Fear und Joy eher mittelmäßig.

➡️ Das deutet auf eine starke Klassen-Ungleichverteilung hin.
Viele seltene Labels könnten durch das Twitter-Modell nur ungenau zugewiesen worden sein.

## 📝 Textbeispiele

Viele Texte sind politisch aufgeladen, extrem lang, stark ideologisch, z. B. Bezug auf Trump, Clinton, Religion, apokalyptische Sprache.

Das Twitter-trained Modell scheint emotionale Labels wie „Anger“ oder „Fear“ aus Emojis und starken Ausdrücken abzuleiten.

Implizite Emotionen oder neutrale Kritik werden oft falsch als „Anger“ oder „Disgust“ klassifiziert.

Viele Texte enthalten Verschwörungsthemen, politische Hashtags und religiöse Rhetorik, die in General-Datasets selten vorkommen.

➡️ Zusammengefasst: Die Labels sind nicht optimal, die daten sind stark einseitig und brauchen eher gradueller aanalysen von wut, zb hate speach, threats,...

## ⚠️ Probleme mit den Labels

Domain Shift: General → Truth Social

Lange Texte: Modell kennt keine langen, komplexen Argumentationen

Ideologische Bias: Texte sind stark politisch; Modell könnte neutralen Text als extrem einstufen

Seltene Labels: Disgust, Sadness, Surprise kaum repräsentiert → schlechtere Klassifikation

#💡 Empfehlungen für bessere Modelle

## Zero-Shot Large Language Models

GPT-5, LLaMA-70B, Mistral-Large

Flexibel bei längeren, ideologisch gefärbten Texten

Labels explizit in Prompt angeben („Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise“)

## Toxic / Hate-Speech Modelle mit Safe-Kategorie

unitary/multilingual-toxic-xlm-roberta

Labels: toxic, severe_toxic, threat, insult, identity_hate, non-toxic

Gut für politische Extreme, Hass, Drohungen

## Fine-Tuning auf Truth Social Daten

Nur ein paar hundert annotierte Beispiele nötig

Verbessert Accuracy für seltene, implizite Labels

Kombiniert mit Zero-Shot kann sehr zuverlässig sein


Beispiel für implizite Emotion oder neutrale Kritik, die falsch als „Anger“ klassifiziert werden könnte:

Text (aus ANGER-Label, Beispiel 3):

"Republicans are not your answer. They are all in it together with the Democrats. Just think about all the stuff they let happen to Trump! Having a few loyal republicans fighting for the people won’t change things. It would take a whole loyal party to make the changes that are needed now! One that is for God, Family, and Our Country First!"

Analyse:

Der Text enthält kritische, politische Analyse, keine direkten aggressiven Ausdrücke oder Hass.

Emotionen sind implizit (Enttäuschung, Sorge, Frustration), nicht explizit Wut.

Das Twitter-Modell interpretiert starke Sprache + Caps + politische Themen als „Anger“.

Tatsächlich könnte man den Text eher als Neutral oder Fear / Concern labeln.

Ein weiteres Beispiel:

Text (FEAR-Label, Beispiel 3):

"Some people say they are struggling to deal with inflation and gas prices and don’t want to hear about the 2020 stolen election. Yes, but it’s on account of that stolen election that you have Biden and the policies that are making your life so much harder #2000Mules"

Der Text ist politisch und kritisierend, enthält aber keine direkte Wut.

Das Twitter-Modell könnte „anger-like“ Elemente (Caps, Hashtags, politische Anklagen) falsch interpretieren.

💡 Fazit:

Implizite Emotionen: Frustration, Sorge, Enttäuschung → oft falsch als Wut/Disgust.

Neutrale Kritik / Analyse → wird durch Caps, politische Begriffe, Hashtags „überdetektiert“.

Wenn du willst, kann ich 5–10 weitere Beispiele aus deinem Dataset markieren, die wahrscheinlich f




ANGER

example 0 : THE DEMOCRATS..BLACK LIVES MATTER ARE ALL ON GODS RADAR<emoji: rage><emoji: fire><emoji: fire>

example 1 : HEAD OF THE CLINTON CABAL ..<emoji: rage><emoji: fire><emoji: fire>

example 2 : EXPOSING THE BIG LIE..THAT SHOULD HAVE PUT A GROUP OF DEMOCRATS IN  PRISON AS TRAITORS<emoji: rage><emoji: fire><emoji: fire>

example 3 : Republicans are not your answer.They are all in it together with the Democrats .Just think about all the stuff they let happen to Trump! Having a few loyal republicans fighting for the people won’t change things . It would take a whole loyal party to make the changes that are needed now !One that is for God,Family,and Our Country First!

example 4 : GOD SAID VENGEANCE IS MINE..I WILL REPAY IN FULL!!!<emoji: rage><emoji: fire><emoji: fire>

example 5 : Why would he do that ? No excuses please. Just facts

example 6 : Fuck this clone!that’s what everyone is thinking In the back Round.what a joke people think this vril clone creature is a president of the free Republic. He’s a fill in clone for the district of Columbia.￼the cabal,￼the VRiL creature queen is dead.let’s play Chess shall we.well I play on the computer level grandmaster 57. For some reason I can’t get to grandmaster 58.For a very long time.when the Queen goes off the bored.The kings next.￼￼￼This officer,has his back turned to him complete disrespe

example 7 : A massive surge of illegal immigrants is right now taking place at our Southern Border - A surge like never seen before and coming in, totally unimpeded. Over 100 countries are “represented.” Our Country is being destroyed!

example 8 : PELOSI AND FBI BEHIND JAN6th..THEY WERE BACK IN SESSION THAT NIGHT!! WHERE IS THE TAPE SHOWING PEOPLE  WALKING PEACEFULLY BETWEEN THE ROPES IN THE ROTUNDA..REPUBLICANS DON’T TEAR DOWN  STATUTES!..BLM SHOULD BE IN PRISON..NO ACCOUNTABILITY !!! GODS JUSTICE WILL PREVAIL!!

example 9 : THATS WHY WE LOOK AND ACT LIKE SODOM AND GOMORRAH.GOD WILL NOT LET THIS GO ON..THIS ONCE  GREAT NATION SOLD ITS SOUL TO THE DEVIL..THESE PRIDE PARADES WITH MEN RIDING NAKED ON BICYCLES IN PORTLAND..PEOPLE BRINGING THEIR CHILDREN TO SEE THIS !!THE WICKED ARE PARADING THEIR SIN..GOD WILL SOON SEND HIS SWORD AND CUT THEM DOWN!!<emoji: fire><emoji: fire><emoji: fire>

DISGUST

example 0 : This country is disgusting. We need trump or a trump affiliate back in office asap. Our current fearless leader belongs in a geriatric center!!! #mytruth #TrumpTruths #SaveAmerica #MAGA

example 1 : Racism, in its definitive meaning, comes in all colors even though the MSM will depict it as belonging only to one!

example 2 : How do I copy this?  I want to post it on fakebook but as you all can imagine - they are allowing NOTHING from Truth to be put on fakebook.  I had five doses of this garbage in the hospital. I want others to know what I didn’t know - at the time!

example 3 : This is spot on. Stop going into a place that has destroyed thousands of small businesses over the decades stop buying that Chinese crap.

example 4 : MAKES SENSE HE’S FULL OF HOT AIR<emoji: fire><emoji: fire>

example 5 : Another bad DJT pick.

example 6 : IN THIS DARK AGE EVERYTHING IS GETTING TWISTED..

example 7 : Look at the eyes. Those are coked out eyes. I have been there. It’s a bad place to be. But he’s hopped up like a mthr fkr

example 8 : Less liberals is not a bad thing<emoji: man-shrugging>

example 9 : Barry is one messed up dude.

FEAR

example 0 : WE NEED TO CALL THEM SATANS SERVANTS..FOR THEY ARE THE CHILDREN OF DARKNESS ..WHO WILL END IN HELL<emoji: fire><emoji: fire>

example 1 : THOUGH THE RIGHTEOUS FALL 7 TIMES THEY RISE AGAIN...BUT THE WICKED IN THIS DARK AGE NOW TO BE BROUGHT DOWN BY CALAMITY.THEIR LAMPS SNUFFED OUT BY THE HAND OF GOD<emoji: fire><emoji: fire>

example 2 : Hey, quick, look over here, the Pentagon released declassified UFO videos,Distraction? #MAGA #MAGAA #magagang #Trump #trumptrains #TrumpTruths #Truth #TRUTHS #TruthSocial

example 3 : Some people say they are struggling to deal with inflation and gas prices and don’t want to hear about the 2020 stolen election. Yes, but it’s on account of that stolen election that you have Biden and the policies that are making your life so much harder #2000Mules

example 4 : YOU ARE SO RIGHT..THE DARK SIDE IS WINNING..TIME FOR THE LIGHT OF THE TRUTH TO EXPOSE ALL THE EVIL<emoji: rage><emoji: fire><emoji: fire>

example 5 : Shape shifters so creepy

example 6 : Tony Bobulinski’s Full interview with Tucker Carlson exposing The Biden crime family. https://rumble.com/v1moa6g-tony-bobulinskis-full-interview-with-tucker-carlson-exposing-the-biden-crim.html

example 7 : #Truth #Covfefe #FBIcorruption #DefundTheFBI

example 8 : Ok does it matter it’s stil Fraud The clone dolly the sheep how many years ago in the 70s 80s What the fuck is up with her eyeball. I’m not trying to make anybody believe anything. I’ve done my research. I’m just throwing out there what I find Researching￼. What I believe 100% 1000% is the truth.And I’m uncovering nothing. As much shit as they were trying to do us  It could feel the Milky Way.￼.￼￼￼With horror￼

example 9 : COUP: General Milley Secretly Pledged to Warn Chinese Communist Party if Trump Planned a Strike.  https://thenationalpulse.com/2021/09/14/coup-general-milley-secretly-pledged-to-warn-chinese-communist-party-if-trump-planned-a-strike/

JOY

example 0 : Special Master has been GRANTED <emoji: us> #MAGA

example 1 : <emoji: us>Happy Independence Day<emoji: us>

example 2 : https://rumble.com/v15oo9e-the-beatles-with-a-little-help-from-my-friend.html

example 3 : Hi! Its all good here! I’ve been traveling a lot. How about you?

example 4 : GONE HOLLYWOOD <emoji: smile><emoji: heart>

example 5 : Thanks for including me in #fbn Follow <emoji: point_right>@FollowBackGirl<emoji: point_left>

example 6 : Well, that pretty much sums it up!

example 7 : @JamiDiChiara @X_Hulley_X AZ is much nicer! Come on over!

example 8 : Thank God for this every day. <emoji: joy>

example 9 : Thank you and have a great Friday!

NEUTRAL

example 0 : TECHNOLOGY THE GREATEST IDOL IN HUMAN HISTORY..MANY ARE BOWING TO..HAVE PEN AND PAPER HANDY AS GOD PREPARES  TO UNPLUG AND TOPPLE IT<emoji: fire><emoji: fire>

example 1 : IN WAYS MANKIND DOES NOT SEE COMING  OR IS PREPARED FOR<emoji: fire><emoji: fire>

example 2 : ADORABLE KOOKABURRA<emoji: heart><emoji: smile>

example 3 : LOVE BALD EAGLES<emoji: smile><emoji: heart><emoji: us><emoji: us>

example 4 : Wake up ... The Best Is Yet To Come !

example 5 : You just can’t make this shit up. <emoji: man-facepalming><emoji: man-facepalming><emoji: man-facepalming><emoji: man-facepalming>

example 6 : If you would love to see Mark Zuckerberg charged with Conspiracy for purposefully rigging the 2020 Presidential Election in favor of Joe Biden, please Re-Truth.

example 7 : GET READY  TO GO TO WAR..THE BATTLE HEATING UP..BUT THIS TIME GOD WILL BE PART OF IT<emoji: fire><emoji: fire>

example 8 : I BET THAT’S A TREASURE TROVE OF DEBAUCHERY<emoji: fire><emoji: fire><emoji: fire>

example 9 : If your vaccine works only if I also take it, your vaccine does not work.

SADNESS

example 0 : conversation has nothing to do with me..needs to be under PAPA

example 1 : PLACE A HEDGE OF PROTECTION AROUND YOUR FAMILY AND THE BLOOD OF CHRIST ON THE DOORFRAMES OF YOUR HOMES.. SO THE ANGEL OF DEATH WILL PASS YOU BY..THIS PANDEMIC IS MERELY A SHOT OVER THE BOW OF THIS WICKED GENERATION..GOD WILL SOON  SEND PLAGUES THAT ARE FAR  MORE DEADLY<emoji: fire><emoji: fire>

example 2 : Translation- Dems are gonna lose in November.

example 3 : PEOPLE ARE WAITING MUCH LONGER TO GET MARRIED ..THEY ARE NOT GOING TO REMAIN CELIBATE TILL  MID THIRTIES.. A CHILD IS A GIFT FROM GOD BUT YOU PRAY IT IS RAISED BY THOSE  WHO WANT THEM….

example 4 : I FOUND CHRIST AFTER I LEFT THE CATHOLIC CHURCH..WE ALL ARE SINNERS SAVED BY GODS GRACE AND CHRISTS BLOOD SHED ON THE CROSS.ASK GOD BETWEEN THE TWO OF YOU TO FORGIVE YOUR SINS. WE ALL HAVE SOME THAT STAND OUT ..INVITE JESUS INTO YOUR HEART AND ASK FOR THE HOLY GUIDE  TO GUIDE YOU..JESUS IS THE ONLY MEDIATOR BETWEEN US AND THE FATHER..WE NEVER NEEDED A PRIEST…GET A  NIV STUDY BIBLE..GET INTO THE NEW TESTAMENT..READ THE PSALMS ..GOD IS LOOKING TO HAVE A RELATIONSHIP WITH US.. BLESSINGS ON YOU<emoji: heart>

example 5 : US Won’t Investigate Governors Who Ordered Nursing Homes to Accept COVID-Positive Residents https://truepundit.com/us-wont-investigate-governors-who-ordered-nursing-homes-to-accept-covid-positive-residents/

example 6 : “It was a mistake.” Colorado CAUGHT Mailing Postcards to Register Illegals for the Midterm Elections  https://nextnewsnetwork.com/2022/10/11/it-was-a-mistake-colorado-caught-mailing-postcards-to-register-illegals-for-the-midterm-elections/

example 7 : A DAY IS A GIFT FROM GOD..NO ONE CAN GIVE THEMSELF.LET US REJOICE AND BE GLAD FOR IT<emoji: heart_decoration><emoji: latin_cross><emoji: heart_decoration>

example 8 : SOON GODS SWORD WILL CUT THE WICKED DOWN..WHEN YOU LOOK FOR THEM THEY WILL BE GONE..THEIR PLACES TO REMEMBER THEM NO MORE ..BUT  THOSE WHO HOPE IN THE LORD WILL INHERIT THE LAND AND DWELL IN IT FOREVER<emoji: heart_decoration><emoji: latin_cross><emoji: heart_decoration>

example 9 : There was someone I spoke with two weeks ago.. From Florida. Sorry if you are not the one

SURPRISE

example 0 : IF GOD HAD NOT PROMISED NOAH HE WOULD NEVER DESTROY THE EARTH BY FLOOD AGAIN ..THIS IS THE GENERATION HE WOULD ERADICATE SO GREAT ARE ITS  SINS IN HIS EYES!!<emoji: fire><emoji: fire>

example 1 : ICYMI EX-RUSSIAN INTEL OFFICER: DEPOPULATION AGENDA IS REAL  http://blog.thegovernmentrag.com/2021/01/18/ex-russian-intel-officer-depopulation-agenda-is-real/

example 2 : REEEEEEEEEEEEEEEEEEEEE

example 3 : GOD IS GOING TO HEAP JUDGEMENT UPON THE TECH GIANTS WHO MADE TECHNOLOGY THE GREATEST IDOL IN HUMAN HISTORY!!I IT WILL TOPPLE LIKE ALL THOSE THAT CAME BEFORE IT<emoji: bangbang>

example 4 : https://creativedestructionmedia.com/video/2022/05/25/breaking-webinar-tonight-live-7pm-est-flccc-releases-covid-vax-injured-treatment-protocol/

example 5 : Whoa <emoji: eyes>

example 6 : Who else thinks Kari Lake is going to make an AMAZING Governor of Arizona? <emoji: man-raising-hand><emoji: us>

example 7 : Unbelievable!! How does she or anyone think this is okay? Appreciate you including me on this one and all of the others!! Follow @cocolicious and @Cmac67 We will follow all back once we are able to.

example 8 : THE WAR IS ONGOING..BUT GOD  HAS NOW ENTERED THE BATTLE..THE VICTORY WILL BE OURS<emoji: us><emoji: us>

example 9 : I POST..GO BACK AND LOOK..WHAT THE HECK!<emoji: smile>


