/*
Copyright (c) 2012 The Go Authors. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
   * Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//This has been translated to c++ from go

#include <cmath>

//Set the following to satisfy your architecture:
//On constants: ULL = UI64, LL = I64, U = UI32, f = F32
typedef int int32;
typedef unsigned int uint32;
typedef long long int64;
typedef unsigned long long uint64;
typedef double float64;
typedef float float32;

const double rn=3.442619855899;

const int64 _LEN=607;
const int64 _TAP=273;
const uint64 _MAX=1ULL<<63;
const uint64 _MASK=_MAX-1;
const int64 _A=48271;
const int64 _M=(1LL<<31)-1;
const int64 _Q=44488;
const int64 _R=3399;

int64 rng_cooked[_LEN]={
	5041579894721019882LL,4646389086726545243LL,1395769623340756751LL,5333664234075297259LL,
	2875692520355975054LL,9033628115061424579LL,7143218595135194537LL,4812947590706362721LL,
	793725219434979937LL,530729988033884841LL,820934885176392507LL,211574159931881404LL,
	459301545753085629LL,814087573554188801LL,331942924126508902LL,861981564819032103LL,
	172707404348361950LL,11310849972103861LL,456951997145934558LL,506283385907531473LL,
	238761877125906442LL,271613134435668611LL,655939277482587688LL,765009320169237031LL,
	768432388404375216LL,25786783599603139LL,659345651940901516LL,27132751497369789LL,
	278938644734011828LL,106519279724614962LL,334450788199935639LL,445979794178006663LL,
	746508166272859988LL,101495080555509718LL,444944072934599077LL,348110936643850264LL,
	241867278911088838LL,579656288757629477LL,448426606444954017LL,373898236197178704LL,
	452359718451235442LL,1053050805812849LL,863383378328234611LL,262530992962879162LL,
	866040596524588430LL,1016283250897194LL,654071468096181739LL,703180231278462085LL,
	624091127734594466LL,83186435546080105LL,800443413754215289LL,211628725166105215LL,
	220230980099216696LL,916102036694505356LL,406929955240776386LL,493638353799262244LL,
	45735150513152492LL,34219504592817935LL,284777168281660050LL,206802011598637651LL,
	436864998958802106LL,88723158709518525LL,556359150688657649LL,681622520025195029LL,
	561697278703408604LL,847180930339483656LL,168657502164118685LL,404548433807426200LL,
	424415621520177892LL,784821733378357738LL,563213652104976190LL,83328314205783527LL,
	902972650836907719LL,324358313466408729LL,431637110180447708LL,893784997996599798LL,
	644694040681043410LL,167934209233237473LL,605063846074242207LL,699352071950958158LL,
	764087785251429360LL,588135342628590798LL,81278655075686088LL,454184558448334333LL,
	272547021627700908LL,498067566014685372LL,521076908060323606LL,889428331899053082LL,
	632644280475008428LL,149581284368424392LL,706975157879912801LL,737025729186023086LL,
	675692927535694226LL,470679451163387365LL,782452046782789866LL,854987509054245321LL,
	3365082947859615LL,132891843575132264LL,729790260180362445LL,101119018391885749LL,
	223802503681785494LL,514715999747391035LL,89651209156052298LL,265947084928637994LL,
	609772935839344860LL,173172598630475368LL,410625584198381271LL,832715521072153550LL,
	847751162068607440LL,580387604467576223LL,843541778086022166LL,598885285665107124LL,
	471583729710395191LL,756617197126448511LL,50580856267889561LL,507009818069506337LL,
	84211066677587151LL,57215682502567780LL,179188101349234089LL,339326709486603876LL,
	377872185047223650LL,235276948318620127LL,129245958384736745LL,889790704367508841LL,
	578180903714416353LL,273395879402949251LL,509201968868075469LL,899612455477252684LL,
	423473717318623208LL,502755828727547283LL,463519858634477230LL,868733889326713935LL,
	590750815073040738LL,78475625547394445LL,97239292751482990LL,542205769480817511LL,
	515842064296928389LL,904853167855864322LL,240721114669887710LL,758328221652109956LL,
	394079651453096228LL,334117463104520637LL,309531388958610294LL,740532189568823871LL,
	583208013294717528LL,789006487514591966LL,818413921079958319LL,114985986140922613LL,
	146459724384021130LL,464164800718799187LL,351649188547146689LL,95628852179165769LL,
	665708996501465751LL,522088435888797935LL,179667732647462064LL,534076197064893291LL,
	114797717161418156LL,506603746554825232LL,257476591183785984LL,108584827984520477LL,
	335010752986839035LL,611643869436655849LL,210770107597129381LL,180329406592126926LL,
	246947805417555887LL,736824328101996598LL,379190836784367752LL,18504697111645663LL,
	225709575651343964LL,721769397107746012LL,90904995307950425LL,719664926854522426LL,
	563766034540086959LL,395554494542796518LL,805752865091741896LL,413926844030112764LL,
	662192658851356805LL,137336113680268144LL,652736623138360001LL,350765457516270089LL,
	920205851277472985LL,195481837689158554LL,664038090713017570LL,829956331917823568LL,
	390186735521895437LL,704631074229557406LL,684719539133399023LL,157263810051886805LL,
	885042267011839972LL,363190914229199290LL,515888109195083128LL,288295831734312159LL,
	476325893181581640LL,628005273434178534LL,424378940820496485LL,204346472802082797LL,
	654530046602208546LL,456258037575859816LL,549545116879542735LL,173831286159015109LL,
	55300461875781649LL,689516063275795982LL,823362392226468517LL,713950633880136085LL,
	855089122238799166LL,553566868813930554LL,243093385335025624LL,540194125786320107LL,
	815964003910772879LL,615749383160077036LL,763206628365814375LL,630832838161710334LL,
	368187876408614036LL,328968613719010974LL,658799720061108684LL,24471477425813547LL,
	407978837741713610LL,809030257594462433LL,294511736343135636LL,86432439584874104LL,
	300903926031262070LL,843002746008253403LL,40108470004599334LL,725462244643869492LL,
	470786415956358861LL,564024853096349395LL,598250771268999789LL,331509824228221010LL,
	550384757877191842LL,394197136717519388LL,811856658030479807LL,383926127401987129LL,
	706241041174209084LL,74138100298020766LL,602799412969025081LL,249782999415006393LL,
	625139033442622883LL,136893024790351883LL,880909639931638024LL,649200435039190070LL,
	246214573746348963LL,40482841892029917LL,415302643423169059LL,26178571525547594LL,
	546471538460007135LL,59271040437876301LL,676412923665775122LL,851365571853935744LL,
	582034366380191420LL,38529852468378991LL,522413500343819946LL,630313164133880214LL,
	715012256130937139LL,36810789914067375LL,311518683455831155LL,291563635358428105LL,
	478258389462771827LL,671829230069998958LL,838708518691437522LL,338751313202475628LL,
	465432937543253823LL,893066756136338160LL,537437343687631927LL,762304235048345395LL,
	772544290181326332LL,918622546756158725LL,409102728959750335LL,235763160649257980LL,
	253093682005861183LL,163655187624004363LL,556466467433496579LL,145224414533431625LL,
	206164238101969082LL,127958026649529403LL,910848158317122100LL,602327868673404980LL,
	500763003267697334LL,215316879295258978LL,672033453496475053LL,604154649113479410LL,
	343392240928378630LL,228547992279730091LL,311061494089657613LL,636655959072284289LL,
	541879141966613650LL,716329841964354375LL,489113805392369699LL,58061851027790701LL,
	168403406525168676LL,442951476735729584LL,33034657855545000LL,111963799581217467LL,
	717751527165346013LL,458904224847080025LL,769328862905900456LL,14360704525844422LL,
	24699430589627362LL,86641732480309928LL,647354711056581607LL,309237993620887689LL,
	205842783951375405LL,513378470852686793LL,878588255630128124LL,614933266684116761LL,
	858584218145447213LL,613767834780551127LL,207044718443697000LL,570822342770557654LL,
	599965789245824450LL,435839141178901242LL,32512300870838984LL,683762169388729092LL,
	484372190531562700LL,601065122214927641LL,539835219896387465LL,460202599011425098LL,
	104464635256904880LL,910661415985316167LL,82925611522859326LL,491928436910299700LL,
	268153255764685089LL,368155947248851187LL,530799951895821403LL,633413038844282927LL,
	265870823291653760LL,116331386505218628LL,58194533750952067LL,364877892071864790LL,
	442367324630654441LL,162079978399695574LL,22082801340951594LL,815038469999938976LL,
	428736051829675300LL,459000018484588384LL,551366085726108518LL,696482910039277427LL,
	47899168835077603LL,874614018568564878LL,22850009133442024LL,135618700745730223LL,
	301925399203419458LL,315260160567850000LL,43015275270600221LL,555958155369697117LL,
	491643298536927566LL,66357493173455439LL,342077383892773207LL,286834862257991557LL,
	199931913404441852LL,332868951863628272LL,258767270978137117LL,151725531352939933LL,
	309234395631736248LL,366225251900706410LL,97244559919649811LL,766486543587595936LL,
	170891353348228256LL,691781716266886849LL,321762902254531290LL,257004302722170710LL,
	873978883954362461LL,248807592462135281LL,469400239538743666LL,455962848179851435LL,
	299720396615329810LL,128255937302635449LL,24011314314667438LL,866571332924651644LL,
	62814133176634675LL,457195081718677047LL,147281118815223540LL,759664802601035582LL,
	609121941775442474LL,783416186482816406LL,710344551887725490LL,439086123735745920LL,
	444265386424057173LL,890348240484733136LL,62226169949417364LL,603726125029721324LL,
	50440494806570911LL,727521552621711306LL,101117678085600140LL,219475010562346106LL,
	262307182861523480LL,515731372807383610LL,373840511196660204LL,253976752407672957LL,
	246728439634926934LL,525602699053685186LL,784108688862839610LL,664085753865589316LL,
	120208733903831749LL,211351499244071597LL,753435089534293140LL,492528473489848474LL,
	514562377147749380LL,822514088013497233LL,271952035438405053LL,913234669781551377LL,
	433215449571016377LL,713778959409434691LL,699472109134426883LL,666722857486904893LL,
	65544004572667749LL,5993474729846685LL,612497402807803640LL,895777478065536541LL,
	233220607194246643LL,170105671228636962LL,315489738361863650LL,163776618138760752LL,
	246052127776757653LL,19730939350268413LL,64367785438526731LL,254317930786193485LL,
	435076901020748511LL,475465208941066767LL,201559550264151451LL,799905945897645860LL,
	428794607148084081LL,836268636677030897LL,648646920932173215LL,361772784584179602LL,
	755435352583430224LL,445002265515354236LL,160519574021353574LL,532701456530550838LL,
	462657581355032832LL,269222202059770514LL,24104557371724986LL,509804697462709401LL,
	791688229546073026LL,88481709029753057LL,532916040953063059LL,779097952885772613LL,
	495507023805937340LL,491853727542267430LL,300807618395040462LL,300776922607115790LL,
	247034623561780302LL,892870277269673173LL,785618792021444590LL,447487458539197488LL,
	790017666060071091LL,214057112791622667LL,242544505726519997LL,248605515334184783LL,
	418667009438202579LL,188393900744603504LL,880866604407486798LL,373413424117847925LL,
	406596887136008919LL,695312420038584778LL,130568681473889905LL,163773909901445764LL,
	365612566094799320LL,396675963463316702LL,310637820408855633LL,632889982277844981LL,
	456538510544025295LL,197988428953949380LL,233179318692086542LL,378320669420892258LL,
	846496120980233608LL,284396375160957768LL,303067819548489632LL,479371757409577260LL,
	445923949480816288LL,40258789580008723LL,805789140871116751LL,454188817093898507LL,
	104266227290881681LL,555730305712256895LL,264767872628324998LL,214447744154983376LL,
	580635221535538708LL,711777100347390362LL,591659717770854163LL,46259771545232136LL,
	883365809702575878LL,597027348142531530LL,56381311938173130LL,276834955065269701LL,
	159882820625087386LL,520639364740355811LL,623504348570926182LL,315221740201463949LL,
	846969326727406649LL,12567292024180741LL,531107962402406093LL,666375493231049158LL,
	873684829504875171LL,448803977499206187LL,592330282348732710LL,14089179108310323LL,
	741494279339357429LL,799042078089695739LL,431781739280707670LL,362518436970536734LL,
	274072276528812270LL,574310000970275834LL,599789864050903915LL,885449334135248416LL,
	524220803543290780LL,70133889989098719LL,760928042919751410LL,302098575511233416LL,
	665132270705551286LL,263519572362116061LL,514452086424602881LL,103508651572782982LL,
	156724209711638904LL,817238926019163658LL,633782035142929227LL,216301256699645892LL,
	274319090289026268LL,190636763322132342LL,601154491566359813LL,593225530735261076LL,
	224112846040631545LL,89550489621669558LL,309448300311137271LL,458385746029296310LL,
	907988717165659497LL,883928918193071140LL,576274038724305787LL,422507205534802623LL,
	183822059838903306LL,380162033680158041LL,882352662008007385LL,177661760558510033LL,
	789905501887764262LL,542167976146300304LL,552110296308627512LL,424827944355936589LL,
	873548753090509853LL,176052709157369297LL,714248504965774589LL,822265687292721812LL,
	496953156492370432LL,339447594219687248LL,642417445326033814LL,35924854507493288LL,
	327365128283173059LL,679710619979713859LL,303091821766509321LL,14560083461731403LL,
	603657585606562623LL,74041625163452715LL,708042763544993558LL,695178137086833547LL,
	39992272236368792LL,29490231444725318LL,784495093633917852LL,88032085863470904LL,
	619265568080867557LL,41160468638471038LL,902680844036512446LL,644078355749758773LL,
	461567463472240429LL,53989729044158054LL,209623822586688385LL,875195563940818268LL,
	190722490805228960LL,738103975730176855LL,615723851339323965LL,774999423191415757LL,
	862957160438089275LL,528043303123908147LL,710161189013981325LL,247901853798576783LL,
	716917692441276957LL,794206649779320330LL,135775972905555768LL,227844743945117484LL,
	362533878574388065LL,647747953900670852LL,897618537557927220LL,551137155471183612LL,
	132602418052089084LL,753744987659604882LL,546468020349969615LL,318967118316219604LL,
	634675175356585710LL,24115998732063030LL,309579344965868205LL,897833284673631015LL,
	290279466227314721LL,720869853019062969LL,727690179233934373LL,173238522931444314LL,
	413329215417082838LL,291830869822419454LL,151946139793714445LL,529393471261659176LL,
	492282895402345266LL,287921153349642564LL,589623639644347210LL,846504381535175242LL,
	732902039687162474LL,891547171701448858LL,294490263567746304LL,705207907349346513LL,
	838214293518882402LL,910392286078035154LL,4152330101494654406LL
};

uint32 kn[128]={
	0x76ad2212U,0x0U,0x600f1b53U,0x6ce447a6U,0x725b46a2U,
	0x7560051dU,0x774921ebU,0x789a25bdU,0x799045c3U,0x7a4bce5dU,
	0x7adf629fU,0x7b5682a6U,0x7bb8a8c6U,0x7c0ae722U,0x7c50cce7U,
	0x7c8cec5bU,0x7cc12cd6U,0x7ceefed2U,0x7d177e0bU,0x7d3b8883U,
	0x7d5bce6cU,0x7d78dd64U,0x7d932886U,0x7dab0e57U,0x7dc0dd30U,
	0x7dd4d688U,0x7de73185U,0x7df81ceaU,0x7e07c0a3U,0x7e163efaU,
	0x7e23b587U,0x7e303dfdU,0x7e3beec2U,0x7e46db77U,0x7e51155dU,
	0x7e5aabb3U,0x7e63abf7U,0x7e6c222cU,0x7e741906U,0x7e7b9a18U,
	0x7e82adfaU,0x7e895c63U,0x7e8fac4bU,0x7e95a3fbU,0x7e9b4924U,
	0x7ea0a0efU,0x7ea5b00dU,0x7eaa7ac3U,0x7eaf04f3U,0x7eb3522aU,
	0x7eb765a5U,0x7ebb4259U,0x7ebeeafdU,0x7ec2620aU,0x7ec5a9c4U,
	0x7ec8c441U,0x7ecbb365U,0x7ece78edU,0x7ed11671U,0x7ed38d62U,
	0x7ed5df12U,0x7ed80cb4U,0x7eda175cU,0x7edc0005U,0x7eddc78eU,
	0x7edf6ebfU,0x7ee0f647U,0x7ee25ebeU,0x7ee3a8a9U,0x7ee4d473U,
	0x7ee5e276U,0x7ee6d2f5U,0x7ee7a620U,0x7ee85c10U,0x7ee8f4cdU,
	0x7ee97047U,0x7ee9ce59U,0x7eea0ecaU,0x7eea3147U,0x7eea3568U,
	0x7eea1aabU,0x7ee9e071U,0x7ee98602U,0x7ee90a88U,0x7ee86d08U,
	0x7ee7ac6aU,0x7ee6c769U,0x7ee5bc9cU,0x7ee48a67U,0x7ee32efcU,
	0x7ee1a857U,0x7edff42fU,0x7ede0ffaU,0x7edbf8d9U,0x7ed9ab94U,
	0x7ed7248dU,0x7ed45faeU,0x7ed1585cU,0x7ece095fU,0x7eca6ccbU,
	0x7ec67be2U,0x7ec22eeeU,0x7ebd7d1aU,0x7eb85c35U,0x7eb2c075U,
	0x7eac9c20U,0x7ea5df27U,0x7e9e769fU,0x7e964c16U,0x7e8d44baU,
	0x7e834033U,0x7e781728U,0x7e6b9933U,0x7e5d8a1aU,0x7e4d9dedU,
	0x7e3b737aU,0x7e268c2fU,0x7e0e3ff5U,0x7df1aa5dU,0x7dcf8c72U,
	0x7da61a1eU,0x7d72a0fbU,0x7d30e097U,0x7cd9b4abU,0x7c600f1aU,
	0x7ba90bdcU,0x7a722176U,0x77d664e5U
};

float32 wn[128]={
	1.7290405e-09f,1.2680929e-10f,1.6897518e-10f,1.9862688e-10f,
	2.2232431e-10f,2.4244937e-10f,2.601613e-10f,2.7611988e-10f,
	2.9073963e-10f,3.042997e-10f,3.1699796e-10f,3.289802e-10f,
	3.4035738e-10f,3.5121603e-10f,3.616251e-10f,3.7164058e-10f,
	3.8130857e-10f,3.9066758e-10f,3.9975012e-10f,4.08584e-10f,
	4.1719309e-10f,4.2559822e-10f,4.338176e-10f,4.418672e-10f,
	4.497613e-10f,4.5751258e-10f,4.651324e-10f,4.7263105e-10f,
	4.8001775e-10f,4.87301e-10f,4.944885e-10f,5.015873e-10f,
	5.0860405e-10f,5.155446e-10f,5.2241467e-10f,5.2921934e-10f,
	5.359635e-10f,5.426517e-10f,5.4928817e-10f,5.5587696e-10f,
	5.624219e-10f,5.6892646e-10f,5.753941e-10f,5.818282e-10f,
	5.882317e-10f,5.946077e-10f,6.00959e-10f,6.072884e-10f,
	6.135985e-10f,6.19892e-10f,6.2617134e-10f,6.3243905e-10f,
	6.386974e-10f,6.449488e-10f,6.511956e-10f,6.5744005e-10f,
	6.6368433e-10f,6.699307e-10f,6.7618144e-10f,6.824387e-10f,
	6.8870465e-10f,6.949815e-10f,7.012715e-10f,7.075768e-10f,
	7.1389966e-10f,7.202424e-10f,7.266073e-10f,7.329966e-10f,
	7.394128e-10f,7.4585826e-10f,7.5233547e-10f,7.58847e-10f,
	7.653954e-10f,7.719835e-10f,7.7861395e-10f,7.852897e-10f,
	7.920138e-10f,7.987892e-10f,8.0561924e-10f,8.125073e-10f,
	8.194569e-10f,8.2647167e-10f,8.3355556e-10f,8.407127e-10f,
	8.479473e-10f,8.55264e-10f,8.6266755e-10f,8.7016316e-10f,
	8.777562e-10f,8.8545243e-10f,8.932582e-10f,9.0117996e-10f,
	9.09225e-10f,9.174008e-10f,9.2571584e-10f,9.341788e-10f,
	9.427997e-10f,9.515889e-10f,9.605579e-10f,9.697193e-10f,
	9.790869e-10f,9.88676e-10f,9.985036e-10f,1.0085882e-09f,
	1.0189509e-09f,1.0296151e-09f,1.0406069e-09f,1.0519566e-09f,
	1.063698e-09f,1.0758702e-09f,1.0885183e-09f,1.1016947e-09f,
	1.1154611e-09f,1.1298902e-09f,1.1450696e-09f,1.1611052e-09f,
	1.1781276e-09f,1.1962995e-09f,1.2158287e-09f,1.2369856e-09f,
	1.2601323e-09f,1.2857697e-09f,1.3146202e-09f,1.347784e-09f,
	1.3870636e-09f,1.4357403e-09f,1.5008659e-09f,1.6030948e-09f
};

float32 fn[128]={
	1.f,0.9635997f,0.9362827f,0.9130436f,0.89228165f,0.87324303f,
	0.8555006f,0.8387836f,0.8229072f,0.8077383f,0.793177f,
	0.7791461f,0.7655842f,0.7524416f,0.73967725f,0.7272569f,
	0.7151515f,0.7033361f,0.69178915f,0.68049186f,0.6694277f,
	0.658582f,0.6479418f,0.63749546f,0.6272325f,0.6171434f,
	0.6072195f,0.5974532f,0.58783704f,0.5783647f,0.56903f,
	0.5598274f,0.5507518f,0.54179835f,0.5329627f,0.52424055f,
	0.5156282f,0.50712204f,0.49871865f,0.49041483f,0.48220766f,
	0.4740943f,0.46607214f,0.4581387f,0.45029163f,0.44252872f,
	0.43484783f,0.427247f,0.41972435f,0.41227803f,0.40490642f,
	0.39760786f,0.3903808f,0.3832238f,0.37613547f,0.36911446f,
	0.3621595f,0.35526937f,0.34844297f,0.34167916f,0.33497685f,
	0.3283351f,0.3217529f,0.3152294f,0.30876362f,0.30235484f,
	0.29600215f,0.28970486f,0.2834622f,0.2772735f,0.27113807f,
	0.2650553f,0.25902456f,0.2530453f,0.24711695f,0.241239f,
	0.23541094f,0.22963232f,0.2239027f,0.21822165f,0.21258877f,
	0.20700371f,0.20146611f,0.19597565f,0.19053204f,0.18513499f,
	0.17978427f,0.17447963f,0.1692209f,0.16400786f,0.15884037f,
	0.15371831f,0.14864157f,0.14361008f,0.13862377f,0.13368265f,
	0.12878671f,0.12393598f,0.119130544f,0.11437051f,0.10965602f,
	0.104987256f,0.10036444f,0.095787846f,0.0912578f,0.08677467f,
	0.0823389f,0.077950984f,0.073611505f,0.06932112f,0.06508058f,
	0.06089077f,0.056752663f,0.0526674f,0.048636295f,0.044660863f,
	0.040742867f,0.03688439f,0.033087887f,0.029356318f,
	0.025693292f,0.022103304f,0.018592102f,0.015167298f,
	0.011839478f,0.008624485f,0.005548995f,0.0026696292f
};

class rngSource {
public:
	int32 tap;
	int32 feed;
	int64 vec[_LEN];
	
	int32 seedrand(int32 x) {
		int32 hi=x/_Q;
		int32 lo=x%_Q;
		x=_A*lo-_R*hi;
		if (x < 0) x+=_M;
		return x;
	}
	
	void rseed(int64 seed) {
		tap=0;
		feed=_LEN-_TAP;

		seed=seed%_M;
		if (seed<0) seed+=_M;
		if (seed==0) seed=89482311LL;

		int32 x=(int32)seed;
		for (int32 i=-20;i<_LEN;i++) {
			x=seedrand(x);
			if (i>=0) {
				int64 u=(int64)x<<40;
				x=seedrand(x);
				u^=(int64)x<<20;
				x=seedrand(x);
				u^=(int64)x;
				u^=rng_cooked[i];
				vec[i]=u&_MASK;
			}
		}
	}
	
	int64 rInt63() {
		tap--;
		if (tap<0) tap+=_LEN;

		feed--;
		if (feed<0) feed+=_LEN;

		int64 x=(vec[feed]+vec[tap])&_MASK;
		vec[feed]=x;
		return x;
	}
	
	uint32 rUint32() {
		return (uint32)(rInt63()>>31);
	}
	
	float64 rFloat64() {
		return (float64)rInt63()/_MAX;
	}
	
	uint32 absInt32(int32 i) {
		if (i<0) return (uint32)(-i);
		return (uint32)i;
	}
	
	float64 rNormFloat64() {
		while(1) {
			int32 j=(int32)rUint32();
			int32 i=j&0x7F;
			float64 x=(float64)j*(float64)wn[i];
			if (absInt32(j)<kn[i]) {
				return x;
			}

			if (i==0) {
				while(1) {
					x=-std::log(rFloat64())*(1.0/rn);
					float64 y=-std::log(rFloat64());
					if (y+y>=x*x) {
						break;
					}
				}
				if (j>0) {
					return rn+x;
				}
				return -rn-x;
			}
			if (fn[i]+(float32)rFloat64()*(fn[i-1]-fn[i])<(float32)std::exp(-.5*x*x)) {
				return x;
			}
		}
	}
};
