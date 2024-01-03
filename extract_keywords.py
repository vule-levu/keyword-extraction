import re
import jieba
import nltk
import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import KMeans
import string
from nltk.corpus import stopwords
from sudachipy import dictionary

nltk.download('stopwords')

# Get English stop words
english_stop_words = stopwords.words("english")

# Get Chinese stop words from jieba
chinese_stop_words = jieba.lcut('')

# Combine all stop words into a single list
stop_words = english_stop_words + chinese_stop_words 

def segment_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    segmented_sentences = []

    for sentence in sentences:
        if all(ord(c) < 128 for c in sentence):
            segmented_sentences.append(sentence.split())
        else:
            segmented_sentences.append(jieba.lcut(sentence))
    cleaned_seg_sentences = [[word.replace('\\n', ' ') for word in sentence] for sentence in segmented_sentences]
    return cleaned_seg_sentences

def tokenize_sentence(sentence):
    # Tokenize a sentence into words using Jieba
    return jieba.lcut("".join(sentence))

def train_word2vec(sentences, vector_size=100, window=5, min_count=1, sg=0, negative=5):
    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, sg=sg, negative=negative)
    model.build_vocab(sentences, update=False)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def train_doc2vec(tagged_documents, vector_size=100, dm=1, min_count=2):
    # Build vocabulary
    model = Doc2Vec(vector_size=vector_size, dm=dm, min_count=min_count)
    model.build_vocab(tagged_documents)

    # Train the model
    model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def calculate_word_similarity(word_vectors):
    # Calculate pairwise cosine similarity between word vectors
    similarities = np.dot(word_vectors, word_vectors.T)
    norms = np.linalg.norm(word_vectors, axis=1)
    similarities /= np.outer(norms, norms)
    return similarities

def cluster_words_by_similarity(word_vectors, num_clusters):
    # Cluster words based on cosine similarity between word vectors
    similarities = calculate_word_similarity(word_vectors)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(similarities)
    return kmeans.labels_, kmeans.cluster_centers_

def calculate_sentence_vector(sentence, word2vec_model, doc2vec_model):
    # Calculate sentence vector by averaging word vectors
    word_vector = [word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv]
    document_vector = doc2vec_model.infer_vector(sentence)
    if word_vector:
        return np.mean(word_vector + [document_vector], axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)


def calculate_word_importance(word_vectors, word_labels, cluster_centers):
    word_importance = {}
    num_clusters = len(cluster_centers)
    cluster_sizes = np.bincount(word_labels, minlength=num_clusters)
    
    for word, vector in zip(word_labels, word_vectors):        
        distances = np.linalg.norm(vector - cluster_centers[word], axis=0)
        distances_sum = np.sum(distances)
        weight = (distances / distances_sum) * cluster_sizes[word]
        word_importance[word] = np.sum(weight)

    return word_importance

def calculate_transition_probability(word_importance, adj_list):
    # Calculate transition probability between nodes
    sum_weights = np.sum([word_importance.get(word, 0) for word in adj_list])
    transition_probability = [word_importance.get(word, 0) / sum_weights for word in adj_list]
    return transition_probability

def build_transfer_matrix(transition_probabilities, word2vec_model):
    # Build transfer matrix
    transfer_matrix = np.zeros((len(word2vec_model.wv.index_to_key), len(word2vec_model.wv.index_to_key)))
    word_to_index = {word: i for i, word in enumerate(word2vec_model.wv.index_to_key)}
    
    for i, adj_list in enumerate(transition_probabilities):
        adj_indices = [word_to_index[word] for word in adj_list if word in word_to_index]
        if adj_indices:  # Check if adj_indices is not empty
            transfer_matrix[i, adj_indices] = transition_probabilities[i]
    
    return transfer_matrix

def remove_chinese_special_characters(word):
    # Remove special characters using regular expression
    word = re.sub(r'[\W\n\t]+', '', word)
    return word

def textrank_keyword_extraction(sentences, topN=5, num_clusters=5, damping_factor=0.85, convergence_threshold=0.0001, max_iterations=100):
        
    # Train Word2Vec model
    word2vec_model = train_word2vec(sentences)

    # Prepare TaggedDocuments for Doc2Vec training
    tagged_documents = [TaggedDocument(words=tokenize_sentence(sentence), tags=[i]) for i, sentence in enumerate(sentences)]

    # Train Doc2Vec model
    doc2vec_model = train_doc2vec(tagged_documents)

    # Calculate sentence vectors
    vector_size = word2vec_model.vector_size
    word_vectors = word2vec_model.wv.vectors
    word_labels, cluster_centers = cluster_words_by_similarity(word_vectors, num_clusters)

    cluster_centers = np.zeros((num_clusters, vector_size))
    for i in range(num_clusters):
        cluster_indices = np.where(word_labels == i)[0]
        cluster_vectors = word_vectors[cluster_indices]
        cluster_centers[i] = np.mean(cluster_vectors, axis=0)

    # Calculate word importance
    word_vectors = word_vectors.reshape(-1, vector_size)  # Reshape vector to a 2D array with a single row
    word_importance = calculate_word_importance(word_vectors, word_labels, cluster_centers)

    # Create adjacency list based on word similarity
    similarity_threshold = 0.5
    adjacency_list = []
    for word in word2vec_model.wv.index_to_key:
        similar_words = [w for w, _ in word2vec_model.wv.similar_by_word(word, topn=len(word2vec_model.wv)) if _ >= similarity_threshold]
        adjacency_list.append(similar_words)

    # Calculate transition probabilities
    transition_probabilities = [calculate_transition_probability(word_importance, adj) for adj in adjacency_list]

    # Build transfer matrix
    transfer_matrix = build_transfer_matrix(transition_probabilities, word2vec_model)

    # Initialize TextRank scores
    initial_scores = np.ones(len(word2vec_model.wv.index_to_key))

    # Iterate to update TextRank scores
    iteration = 0
    while True:
        new_scores = update_textrank_scores(transfer_matrix, damping_factor, initial_scores)
        diff = np.sum(np.abs(new_scores - initial_scores))

        if diff < convergence_threshold or iteration >= max_iterations:
            break

        initial_scores = new_scores
        iteration += 1

    # Sort the words based on TextRank scores
    sorted_word_indices = sorted(range(len(word2vec_model.wv.index_to_key)), key=lambda i: initial_scores[i], reverse=True)
    sorted_words = [word2vec_model.wv.index_to_key[i] for i in sorted_word_indices]

    # Get the top N keywords
    top_keywords = []
    for word in sorted_words:
        word = remove_chinese_special_characters(word)
        if (len(word) > 1 and word.isalpha()) or (len(word) == 1 and not word.strip()) and word not in string.punctuation and word not in stop_words:
            top_keywords.append(word)
        if len(top_keywords) == topN:
            break

    return top_keywords


def update_textrank_scores(transfer_matrix, damping_factor, initial_scores):
    n = transfer_matrix.shape[0]
    e = np.ones(n)
    new_scores = ((1 - damping_factor) * e / n) + (damping_factor * transfer_matrix.T @ initial_scores)
    return new_scores


text = "这是一个示例句子。另一个用于演示的句子。TextRank是一种基于图的算法。要法キマウレ説伴サヘ医宅だいドせ禁地レ期8専シ暖間ネカモ党投継リ持要ぐ視司ラ拾沢に情質んぎえは冬京厘む。骨文ば併学チヘタ反周紙ハ始度時ぴべ謙後83区ぶげクで臨稿ヤ嶋止フヘ日勇隆早い。熊ホラエ的他ヒワ伝備ぜく陽悩併ドもむ提話はす成85意りけ窓決ま題春イて献同惨だね不姿カナケ格闘ネハル当治マホナヌ均重つ認事ホヨエ大商溢イぞむみ勝丼冨でむ。出鳥べ改発なで談眺ヱタイホ油著ヒフナミ神給ヘノ平馬ぜ有気ぽこ予報ライオコ残63可渡ぐ際守カ康文よ技檀ユ強生ト何員やじそラ備略教レに動対具察埋リんせて。文ヤサニ県7次っろレ行材ば闘姿責4制ばゃ最問健シワマ地則少シ和描ちつわみ田茶崎さトゆた総県ル先県イ立再ぎク乗命び。基主ソ神夫てイとづ満客極社さ第聞がっづね速記94山ドれさレ誓性ルヨハ和増ひもよ歳強ヲカ決宣獲ゃろレだ。百ヒルラ芸3要御クマ頼雨レをト外問チヒ金文え拡員べふ禁飛いもねび任徳ツ手簸が止権へが花孝研うこッ。題へぽぐょ情今ぜをぴ石権件カキモオ読稿ーへトゃ外時さまほは公65自ヘトサニ試理ミヨル着頃にのえ必暴新ナ択試ぴに。読キロユヲ乗致じれぴ災任えれけイ北歯ワア民新アヤ君禁ワ目45発むリ座標なかきど埼8持ホイニ続照ね済8済19人監都成ぜさ。人4載づう世戊トナレマ評百定さラだ茶学コルミ倉額タヘスカ市長与イ載暮ぞち定認欺キコミ情退ぎルレ加爺議ぜスぐ答仙に載品ス喜目テシヒニ玉限どうとぞ。局ロツレニ鮮投ごせる棋権こだき関祭ハメコキ易臨ぽぶも地輔やの康車ツヱ場道フせクト波供ヘテカ郵聞近独クずりお監到敷砂翼ずせいル。63勲悔殿4引ミヘ王海ぴぐいル菜化げクあす社用億ハ整覧記マユタレ未言エネニ銀写レだ理読ワ通2躍ぼ派田ホ農疑ラ断円常ミラタチ本偶凝峡さぐ。当ぞリてぱ総崎ほべクす望舟す蔵車ぴそざラ幅観交えまわ写放ラワ真風せ円名クぴぜ億式ばけだじ中扱リルハツ中供ふ昭面委ゅつげふ。趙権写ゆりスわ社倉ヱアマソ聞提始ぽも済80会象ず山著メラ健方タ本源ケマクモ販愛ぱ表材委そド急稿トマムナ攻囚廉てけ。済トカサ由協リツチホ断角ご子供ルなざリ海開リホ活席サク学熊あこ県毎とりあ端段あきまど法育にぴお要催ワトハム小身ーてんっ四囚廉スぐけ。本るにびろ帯権百コムノ通銀じんッ報写スめ政自ッぱのゆ作込んふゅの容花クタ連法ムネニラ捕属告ざき断題別数ナヱオウ送出ぞッ瑣9下セシコソ見析ぐ。25輝ヤ視芸ス徳9員てげぼつ愛聞政ネマ会助づゅでよ野治写ぎかドイ祭成透ヱユネム者10福尊森ミ答建ヌケヒク女戒会種児け。同図さ党楽ご粧動銃ヌリ横日リ測感ワセエチ聞森ぽ知業どみ給発ねう町位十フ又9場ゅどへ読偵フ案認ワヘカ量伸換負しレこあ。半に催空ヤム胴向60庭ネイヒチ募帯し機著ホ日6仕フル止員形セサテ戦厳チキヲリ仕水シ飲適あば写腹あ可梁ッほそル問素敷隊誇誤う。記ワハクロ大事池ソニ図場ぞ企64弁伝0沿ぐ進意ふょとぴ製経間ヒ検特モリル歳菱かたリに機曽地リヌキ記后嘘姉ゃきべぎ。月康報どみべく医導エスオカ花吉マタチヌ視活らわン青着うかぶを備問ヤレユ案断そぴのー覧指雪ヌエヒラ賞天件げ念覧フね差択らうれ年県トび験台はほ社就キニス児2紙ユノヌ権拘肌陶床て。席75案ニ全表に殊貨機潟ホモ銃8球スい篇給ソムヒラ今変もゅへれ歳就ハタ同6式3聞キミセ酔力と校際ッと学埋詰よっぎ。衝どねづく授法コリ乗再イ称閉ヨ昌族ト著求みごぎ会革69謙ム報昔りょリ影後ともこ制放夕死テルホ糧購ぞラわ絶郎ヒミ格第ルむぜ文障置ヱツウヒ南新六仕足ぎずーぽ。堂アヤニロ情胸要ふぶイ位改ヤミマ証2説ばら産質レコメヤ内立家ふ超説レさラい修点ロリオ受的6道ごし倉軽こぼ歴下ふま阪並28死の。敵なの戒能ぜろ根立付ケコフ栃劇覇ラさぱ能哲カ長需評エチタ鋭傷サアヌ画足っしぽぱ意局ざ帯放ハヒヤコ年富員記ーしき。治ぴつ人国ロヒモシ銅正ハヘユチ出子らぱす謀四うば将五も宿中原え妹図ノエヒヘ製暮チユ玉断イロテリ回1金ぽぞく東楽シユケキ田5者聞ミ棋問で速載きぽほ町拘肌陶わ。万ヱツナ生58止触んのづド中異ず社74均拠池鮮7属認チ禁援ずりげ難当はゅ史開えみフゃ性置フツイ方経合禁ぜご。Lorem ipsum dolor sit amet, consectetur adipiscing elit. Post enim Chrysippum eum non sane est disputatum. Istam voluptatem perpetuam quis potest praestare sapienti? Ad eas enim res ab Epicuro praecepta dantur. Haec bene dicuntur, nec ego repugno, sed inter sese ipsa pugnant. Velut ego nunc moveor. Duo Reges: constructio interrete.Bork In his igitur partibus duabus nihil erat, quod Zeno commutare gestiret. An hoc usque quaque, aliter in vita? In primo enim ortu inest teneritas ac mollitia quaedam, ut nec res videre optimas nec agere possint. Praetereo multos, in bis doctum hominem et suavem, Hieronymum, quem iam cur Peripateticum appellem nescio. Itaque haec cum illis est dissensio, cum Peripateticis nulla sane. Ut non sine causa ex iis memoriae ducta sit disciplina. Cur tantas regiones barbarorum pedibus obiit, tot maria transmisit? Universa enim illorum ratione cum tota vestra confligendum puto. Id Sextilius factum negabat. Claudii libidini, qui tum erat summo ne imperio, dederetur. Voluptatem cum summum bonum diceret, primum in eo ipso parum vidit, deinde hoc quoque alienum;習采蝴草打入筆朋房黃頭刀昔快媽寺蛋道；兩家歌共玩亮去國園做斤再東實草訴，己冰壯聲十抓免香刃會耳重停母雪。水鳥包年丁物着住豆，很心刀采。知一棵筆尼好草房？回穿尾羊欠屋竹下同樹面戶浪帽坐故書福，學果朱書室哥蝸足民直老：冰大相生父卜菜浪蝸掃去玩哪七。功十耳泉寸言菜明。菜禾田眼松圓抱嗎歌躲，天竹尤寺土歌欠叫開光示童們帶心鴨尤河，肉哭品安根目，根把植人花什根外祖地背跑錯候爸旁紅亭文采！登半去干哭里星升點路課出男牛忍，竹安員對。路羊動筆布司完百圓說看身身教手怕戶。升話尾雲象再目外起雨發娘司肖上：了黑元左媽犬皮買央眼服，背目夏拍合進歌午兩念皮陽後欠紅立，裝未己。主園借丟或小。巴食大黑念間知別蝴早足裝母發。色明秋欠種星息嗎胡早們兩問躲夕邊院；固讀杯親太許收常國但，早交六兔，邊風石兄玉從免飽吉要央眼南虎到瓜旁，者海夏意風日鳥三躲皮連拍占虎「毛常勿話再小地中」良。雲昔衣八而功太生害具娘六吧澡不；禾壯刃去看吃想右在壯蛋干日什頭呀汗再有！今流裝，神土白玉次登喜坡具犬兔許枝家但收苦。也員詞未刃要助牙教虎走北朋急圓乾遠采回。話內道世女故前足的首山故消親音古央詞右。昔都抱干力蝸荷下昔去比斗！定科呢由路找加；丟杯丁。"
sentences = segment_text_into_sentences(text)
N = 20  # Number of top keywords you want to retrieve
top_N_keywords = textrank_keyword_extraction(sentences, N)
print("keywords")
print(top_N_keywords)
