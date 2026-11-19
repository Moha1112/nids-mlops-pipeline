import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Charger la cle GOOGLE_API_KEY depuis le fichier .env
load_dotenv()

# 1. Base de connaissances Cybersecurite (Documents LangChain MITRE ATT&CK)
MITRE_DOCS = [
    Document(
        page_content=(
            "Attack Type: Port Scanning. MITRE ATT&CK ID: T1046 (Network Service Discovery). "
            "Description: Balayage de ports (ex: Nmap) pour decouvrir les services ouverts (SSH, HTTP, SMB). "
            "Remediation: Bloquer l'IP source via iptables/pare-feu, activer Fail2Ban, fermer les ports inutiles."
        ),
        metadata={"attack_type": "Port Scanning", "mitre_id": "T1046"}
    ),
    Document(
        page_content=(
            "Attack Type: DDoS. MITRE ATT&CK ID: T1498 (Network Denial of Service). "
            "Description: Saturation de la bande passante ou des ressources par un volume massif de paquets. "
            "Remediation: Activer le Rate Limiting, activer la protection anti-DDoS (Cloudflare/WAF), filtrer les SYN flood."
        ),
        metadata={"attack_type": "DDoS", "mitre_id": "T1498"}
    ),
    Document(
        page_content=(
            "Attack Type: DoS. MITRE ATT&CK ID: T1499 (Endpoint Denial of Service). "
            "Description: Epuisement des ressources applicatives (Hulk, Slowloris, GoldenEye). "
            "Remediation: Limiter les connexions simultanees par IP, reduire le timeout HTTP Keep-Alive, bannir l'IP dans le WAF."
        ),
        metadata={"attack_type": "DoS", "mitre_id": "T1499"}
    ),
    Document(
        page_content=(
            "Attack Type: Brute Force. MITRE ATT&CK ID: T1110 (Brute Force). "
            "Description: Tentatives automatisees de connexion (SSH-Patator, FTP-Patator) pour deviner des mots de passe. "
            "Remediation: Verrouiller le compte cible, desactiver le login par mot de passe SSH (utiliser des cles), imposer le MFA."
        ),
        metadata={"attack_type": "Brute Force", "mitre_id": "T1110"}
    ),
    Document(
        page_content=(
            "Attack Type: Web Attack. MITRE ATT&CK ID: T1190 (Exploit Public-Facing Application). "
            "Description: Attaque applicative web (SQL Injection, XSS, Web Brute Force). "
            "Remediation: Activer OWASP Core Rule Set sur le WAF, isoler le serveur web, auditer les logs de la base de donnees."
        ),
        metadata={"attack_type": "Web Attack", "mitre_id": "T1190"}
    ),
    Document(
        page_content=(
            "Attack Type: Bots. MITRE ATT&CK ID: T1071 (Application Layer Protocol - Botnet C2). "
            "Description: Communication periodique entre un poste infecte et un serveur Commande et Controle (C2). "
            "Remediation: Isoler la machine dans un VLAN de quarantaine, bloquer le domaine/IP C2 sur le DNS, lancer un scan EDR."
        ),
        metadata={"attack_type": "Bots", "mitre_id": "T1071"}
    ),
    Document(
        page_content=(
            "Attack Type: Normal Traffic. MITRE ATT&CK ID: N/A. "
            "Description: Trafic reseau legitime sans anomalie detectee. "
            "Remediation: Aucune action requise."
        ),
        metadata={"attack_type": "Normal Traffic", "mitre_id": "N/A"}
    ),
]

class SocCopilotRAG:
    def __init__(self):
        # Embeddings & Base vectorielle FAISS
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        self.vectorstore = FAISS.from_documents(MITRE_DOCS, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 1})

        # Modele de langage (LLM)
        self.llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

        # Prompt d'analyste SOC
        self.prompt = ChatPromptTemplate.from_template(
            """Tu es un analyste senior en cybersecurite (SOC).
En te basant uniquement sur le contexte MITRE ATT&CK suivant, redige un rapport d'incident court et actionnable en francais.

Contexte MITRE ATT&CK :
{context}

Alerte detectee par le modele XGBoost :
{question}

Reponds avec exactement 3 sections courtes :
1. Identification (ID MITRE ATT&CK et gravite)
2. Analyse de la menace
3. Plan de remediation immediat (commandes ou actions precises)"""
        )

        # Chaine RAG moderne avec LCEL
        self.chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def explain_alert(self, attack_type: str, confidence: float, dest_port: float = 0.0) -> str:
        query = f"Attaque: {attack_type} | Confiance du modele: {confidence:.2%} | Port destination: {int(dest_port)}"
        return self.chain.invoke(query)

# Test rapide en ligne de commande
if __name__ == "__main__":
    copilot = SocCopilotRAG()
    print(copilot.explain_alert("Port Scanning", 0.984, dest_port=22))