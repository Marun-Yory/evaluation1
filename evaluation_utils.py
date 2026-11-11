
import json
import time

from tqdm.auto import tqdm
from rag_helper import RAGBase


def calc_price(usage):
    input_price_per_million = 0.10
    output_price_per_million = 0.40

    input_cost = (
        usage.input_tokens / 1_000_000
    ) * input_price_per_million

    output_cost = (
        usage.output_tokens / 1_000_000
    ) * output_price_per_million

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def calc_total_price(usages):
    return sum(
        calc_price(usage)["total_cost"]
        for usage in usages
    )


def llm_structured(
    client,
    instructions,
    user_prompt,
    output_type,
    model="qwen-max",
):
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]

    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=output_type,
    )

    return response.output_parsed, response.usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="openai/gpt-oss-120b",
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client=client,
                instructions=instructions,
                user_prompt=user_prompt,
                output_type=output_type,
                model=model,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise

            time.sleep(2 ** attempt)


class CurrencyRAG(RAGBase):

    def __init__(
        self,
        *args,
        currency_file="currencies.json",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.currency_file = currency_file
        self.usages = []
        self.last_usage = None

        with open(
            self.currency_file,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        self.currencies = data["currencies"]

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        # Search fields should match the currency dataset.
        boost_dict = {
            "code": 3.0,
            "name": 2.0,
            "symbol": 2.0,
            "countries": 2.0,
            "decimal_digits": 0.5,
        }

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
        )

    def search_currency_by_code(self, code):
        code = code.upper()

        return [
            currency
            for currency in self.currencies
            if currency["code"] == code
        ]

    def search_currency_by_country(self, country):
        country = country.strip().casefold()

        return [
            currency
            for currency in self.currencies
            if any(
                name.casefold() == country
                for name in currency["countries"]
            )
        ]

    def search_currency_by_name(self, name):
        name = name.strip().casefold()

        return [
            currency
            for currency in self.currencies
            if name in currency["name"].casefold()
        ]

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages,
        )

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.output_text

    def total_cost(self):
        return calc_total_price(self.usages)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for element in seq:
            future = pool.submit(f, element)
            future.add_done_callback(
                lambda _: progress.update()
            )
            futures.append(future)

        for future in futures:
            results.append(future.result())

    return results
