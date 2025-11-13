import json
import joblib
import re
from collections import defaultdict, Counter
from wordfreq import zipf_frequency
from typing import List
import psycopg
import httpx
from model import dtype


class HybridPersonalizedModelPG:
    def __init__(self, model_path="expense_classifier_pa.pkl", vectorizer_path="vectorizer_tfidf_pa.pkl", encoder_path="label_encoder_pa.pkl"):
        self.db_config = {
            "dbname": "postgres",
            "user": "postgres",
            "password": "12345",
            "host": "localhost",
            "port": 5432
        }

        self.X_model = joblib.load(model_path)
        self.X_vectorizer = joblib.load(vectorizer_path)
        self.X_label_encoder = joblib.load(encoder_path)

        self.NER_vectorizer = joblib.load("vectorizer_tfidf.pkl")
        self.NER_model = joblib.load("expense_classifier_xgb.pkl")
        self.le = joblib.load("label_encoder.pkl")

        self.conn = None

    async def main(self, url, user_id):
        if not user_id:
            return None

        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code == 200:
                return response.json()

        return None

    async def process(self, records, categories, user_id):
        all_descriptions, money, date, results = [], [], [], []
        grouped, json1 = defaultdict(list), []

        if isinstance(records, dict):
            records = [records]

        for record in records:
            for account in record.get("Accounts", []):
                for t in account.get("Transactions", []):
                    desc = t.get("Description", "").replace("\n", " ")
                    mny = t.get("WithdrawalAmount") or t.get("DepositAmount")
                    dte = t.get("Date")
                    if desc:
                        all_descriptions.append(desc)
                        money.append(mny)
                        date.append(dte)

        predictions = []
        confid = []
        for decs in all_descriptions:
            predicted_label = await self.predict(user_id, [decs])
            suggestion = predicted_label[0] if predicted_label[0] in categories else "Others"
            predictions.append(suggestion)
            confid.append(predicted_label[1])

        orgs = []
        for text in all_descriptions:
            txt = re.findall(r"[A-Za-zÀ-ÿ]+", text)

            texts = self.NER_vectorizer.transform([" ".join(txt)])
            preds = self.NER_model.predict(texts)
            org = self.le.inverse_transform(preds)[0]
            print(org)
            orgs.append(org)

        org_suggestion_count = defaultdict(lambda: {'suggestions': Counter(), 'total': 0})

        for org, suggestion in zip(orgs, predictions):
            org_suggestion_count[org]['suggestions'][suggestion] += 1
            org_suggestion_count[org]['total'] += 1


        org_to_cats = defaultdict(list)
        for org, cat in zip(orgs, predictions):
            if org != "Others":
                org_to_cats[org].append(cat)

        org_top_suggestion = {}
        org_confidence = {}
        for org, cat_list in org_to_cats.items():
            cat_counts = Counter(cat_list)

            top_cat, top_count = cat_counts.most_common(1)[0]
            total = sum(cat_counts.values())
            confidence = round((top_count / total) * 100, 2)

            org_top_suggestion[org] = top_cat
            org_confidence[org] = confidence

        for i, decs in enumerate(all_descriptions):
            org = orgs[i]

            suggestion = (
                org_top_suggestion.get(org, predictions[i])
                if org != "Others"
                else predictions[i]
            )

            data = {
                "Date": date[i],
                "Description": decs,
                "Amount": money[i],
                "Organization": orgs[i],
                "Suggestion": suggestion,
                "Confidence": confid[i]
            }

            json1.append(data)

            if org != "Others":
                value = dtype(org, money[i], suggestion)
                results.append(value)
       
            else:
                value = dtype(org, money[i], suggestion)
                results.append(value)

        for item in results:
            grouped[item.Organization].append(item.Amount) 

        json2 = []
        i = 0
        for grp, cash_list in grouped.items():
            org_count = org_suggestion_count.get(grp, {'suggestions': Counter(), 'total': 0})

            info = {
                "Organization": grp,
                "Occurence": len(cash_list),
                "Amount": sum(cash_list),
                "Category": org_top_suggestion.get(grp, "Others"),
                "Total_Transactions": org_count['total'],
                "Suggestions_Count": dict(org_count['suggestions'])
                }
            i = i + 1
            json2.append(info)

        if data:
            print("Descriptions:", len(all_descriptions))
            print("Final categories:", len(data))
            print(json.dumps(json1, indent=5))
            print(json.dumps(json2, indent=5))

        # return json1, json2

    async def _connect_db(self):
        if not self.conn:
            self.conn = await psycopg.AsyncConnection.connect(**self.db_config)
            await self.conn.set_autocommit(True)

        return self.conn

    async def _save_user_memory_entry(self, user_id, text, label, org, cat_count, conf):
        conn = await self._connect_db()

        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO user_memory (user_id, text, category, org_name, cat_count, confidence) VALUES (%s, %s, %s, %s, %s, %s)",
                (user_id, text, label, org, cat_count, conf)
            )

    async def predict(self, user_id, text: List[str]):
        conn = await self._connect_db()

        text = text[0]

        txt = re.findall(r"[A-Za-zÀ-ÿ]+", text)

        texts = self.NER_vectorizer.transform([" ".join(txt)])
        preds = self.NER_model.predict(texts)
        org = self.le.inverse_transform(preds)[0]

        async with conn.cursor() as cur:
            await cur.execute("SELECT category, confidence FROM user_memory WHERE user_id = %s AND org_name = %s", (user_id, org))
            data = await cur.fetchall()

        if data != []:
            largest = max(data, key=lambda x: x[1])
            return largest

        X_sample = self.X_vectorizer.transform([text])
        pred = self.X_model.predict(X_sample)
        label = self.X_label_encoder.inverse_transform(pred)[0]
        predics = (label, 0)

        return predics

    async def feedback(self, user_id):
        if not user_id:
            return None

        conn = await self._connect_db()

        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT text, category, org_name FROM user_table WHERE user_id = %s",
                (user_id,)
            )
            data = await cur.fetchall()

        rows = []
        for text, lbl, org in data:
            org = org.strip()
            clean_text = re.sub(r'[^a-z\s]+', '', text.lower())
            clean_text = ' '.join(w for w in clean_text.split() if zipf_frequency(w, 'en') >= 2.0)
            rows.append((clean_text, lbl, org))

        if not rows:
            print(f"[INFO] No labeled data found for user {user_id}.")
            return "no data"

        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM user_memory WHERE user_id = %s", (user_id,))

        org_suggestion_count = defaultdict(lambda: {'suggestions': Counter(), 'total': 0})
        for txt, suggestion, org in rows:
            org_suggestion_count[org]['suggestions'][suggestion] += 1
            org_suggestion_count[org]['total'] += 1

        async with conn.cursor() as cur:
            for org, stats in org_suggestion_count.items():
                cat_dict = stats['suggestions']
                total = stats['total']

                for category, count in cat_dict.items():
                    confidence = (count / total * 100) if total > 0 else 0

                    await cur.execute(
                        """SELECT id, cat_count FROM user_memory 
                        WHERE user_id = %s AND org_name = %s AND category = %s""",
                        (user_id, org, category)
                    )
                    existing = await cur.fetchone()

                    if existing:
                        record_id, old_count = existing
                        new_count = old_count + count
                        new_conf = (new_count / total * 100) if total > 0 else 0
                        await cur.execute(
                            """UPDATE user_memory 
                            SET cat_count = %s, confidence = %s 
                            WHERE id = %s""",
                            (new_count, new_conf, record_id)
                        )
                    else:
                        texts = [t for t, c, o in rows if c == category and o == org]
                        joined_text = " ".join(texts)

                        await self._save_user_memory_entry(
                            user_id, joined_text, category, org, count, confidence
                        )

        return {"updated": "Success"}

    async def feedback_signal(self, user_id):
        if not user_id:
            return None

        conn = await self._connect_db()

        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, text, category FROM user_table WHERE user_id = %s",
                    (user_id,)
                )

                rows = await cur.fetchall()
                if not rows:
                    return f"No records found against id: '{user_id}'"

            return user_id

        except Exception as e:
            return str(e)
