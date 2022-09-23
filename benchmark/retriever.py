logger = logging.getLogger(__name__)


class RocketQARetriever(BaseRetriever):
    def __init__(
        self,
        document_store: BaseDocumentStore,
        query_embedding_model: Union[Path, str] = "facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model: Union[Path, str] = "facebook/dpr-ctx_encoder-single-nq-base",
        model_version: Optional[str] = None,
        max_seq_len_query: int = 64,
        max_seq_len_passage: int = 256,
        top_k: int = 10,
        use_gpu: bool = True,
        batch_size: int = 16,
        embed_title: bool = True,
        use_fast_tokenizers: bool = True,
        similarity_function: str = "dot_product",
        global_loss_buffer_size: int = 150000,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        scale_score: bool = True,
    ):
        super().__init__()

        self.document_store = document_store
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.top_k = top_k
        self.scale_score = scale_score

        # Load model
       #from config import DOCQA_DE_CONF
       #self.dual_encoder = rocketqa.load_model(**DOCQA_DE_CONF)

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = None,
    ):
        if top_k is None:
            top_k = self.top_k
        if not self.document_store:
            logger.error("Cannot perform retrieve() since DensePassageRetriever initialized with document_store=None")
            return []
        if index is None:
            index = self.document_store.index
        if scale_score is None:
            scale_score = self.scale_score

        # 1. Query -> emb
        q_embs = self.embed_queries(query)
        # 2. Emb -> Search -> Docs
        documents = self.document_store.query_by_embedding(
            query_emb=q_embs[0], top_k=top_k, filters=filters, index=index, headers=headers, scale_score=scale_score
        )

        return documents

    def retrieve_batch(
        self,
        queries: List[str],
        filters = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: bool = None,
    ):
        logger.error("Not implemented!")
        return None


    def embed_queries(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        assert isinstance(texts, list), "Expecting a list of texts when embed queries"

        q_embs = list()
        for tt in texts:
          q_embs.append(encode_query(tt))

        return np.array(q_embs)

    def embed_documents(self, docs):
        embs = list()
        for doc in docs:
          para = doc.content
          embs.append(encode_para(para))

        return np.array(embs)


    def _embed_queries(self, texts):
        """
        Create embeddings for a list of queries using the query encoder
        List[str] -> List[np.ndarray]

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        if isinstance(texts, str):
            texts = [texts]
        assert isinstance(texts, list), "Expecting a list of texts when embed queries"
        q_embs = self.dual_encoder.encode_query(query=texts)
        q_embs = np.array(list(q_embs))

        return q_embs

    def _embed_documents(self, docs):
        """
        Create embeddings for a list of documents using the passage encoder
        List[Document] -> List[np.ndarray]

        :param docs: List of Document objects used to represent documents / passages in a standardized way within Haystack.
        :return: Embeddings of documents / passages shape (batch_size, embedding_dim)
        """
        para_list = [ d.content for d in docs ]
        title_list = []
        para_embs = self.dual_encoder.encode_para(
            para=para_list, title=title_list)
        para_embs = np.array(list(para_embs))
