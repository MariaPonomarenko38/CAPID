import random
import string
import random
from datetime import datetime, timedelta
import random
import string
from capid.utils import call_openai
from faker import Faker

fake = Faker()

class CategoryBase:

    def __init__(self, context):
        self.context = context

class Code(CategoryBase):

    def __init__(self, context):
        self.context = context
        self.last_method = None

    def social_security_number(self):
        return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"

    def drivers_license(self):
        letters = ''.join(random.choices(string.ascii_uppercase, k=1))
        numbers = ''.join(random.choices(string.digits, k=8))
        return letters + numbers

    def bank_account(self):
        length = random.choice([10, 12])
        return ''.join(random.choices(string.digits, k=length))

    def credit_card(self):
        return ''.join(random.choices(string.digits, k=16))

    def phone_number(self):
        return f"+{random.randint(1,9)}{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"

    def ip_address(self):
        return ".".join(str(random.randint(0, 255)) for _ in range(4))

    def email_address(self):
        user = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        domain = random.choice(["gmail", "yahoo", "outlook", "protonmail", "hotmail"])
        tld = random.choice(["com", "org", "net"])
        return f"{user}@{domain}.{tld}"

    def password_hash(self):
        return ''.join(random.choices(string.hexdigits.lower(), k=64))

    def passport_number(self):
        letters = ''.join(random.choices(string.ascii_uppercase, k=1))
        numbers = ''.join(random.choices(string.digits, k=8))
        return letters + numbers

    def tax_id(self):
        return ''.join(random.choices(string.digits, k=9))

    def employee_id(self):
        letters = ''.join(random.choices(string.ascii_uppercase, k=1))
        numbers = ''.join(random.choices(string.digits, k=6))
        return letters + numbers

    def student_id(self):
        return ''.join(random.choices(string.digits, k=8))

    def generate_random(self):
        methods = [
            self.social_security_number,
            self.drivers_license,
            self.bank_account,
            self.credit_card,
            self.phone_number,
            self.ip_address,
            self.email_address,
            self.password_hash,
            self.passport_number,
            self.tax_id,
            self.employee_id,
            self.student_id
        ]
        method = random.choice([m for m in methods if m.__name__ != self.last_method])
        self.last_method = method.__name__
        return " ".join(method.__name__.split("_")), method()

class Datetime:
    def __init__(self, context):
        self.context = context
        self.last_method = None

    def date(self):
        formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%b %d, %Y",       
            "%d %b %Y"        
        ]

        random_days_ago = random.randint(-20000, 20000)
        dt = datetime.now() + timedelta(days=random_days_ago)

        return dt.strftime(random.choice(formats))

    def time(self):
        formats = [
            "%H:%M",           
            "%H:%M:%S",        
            "%I:%M %p",       
            "%I:%M:%S %p"      
        ]
        tz = random.choice(["", " UTC", " EST", " PST", " CET", " GMT"])
        return datetime.now().strftime(random.choice(formats)) + tz

    def duration(self):
        style = random.choice(["single", "double", "long"])

        if style == "single":
            return f"{random.randint(1, 200)}{random.choice(['d','h','m','s'])}"

        if style == "double":
            return f"{random.randint(1, 48)}h {random.randint(1,59)}m"

        if style == "long":
            return f"{random.randint(1,10)}d {random.randint(1,23)}h {random.randint(1,59)}m"

    def generate_random(self):
        methods = [self.date, self.time, self.duration]
        method = random.choice([m for m in methods if m.__name__ != self.last_method])
        self.last_method = method.__name__
        return " ".join(method.__name__.split("_")), method()

class Finance:
    def __init__(self, context):
        self.context = context
        self.last_method = None

    def _currency_value(self, min_val=0, max_val=500000):
        currency = random.choice(["$", "€", "£"])
        amount = random.randint(min_val, max_val)
        return f"{currency}{amount}"

    def monthly_income(self):
        return self._currency_value(500, 20000)

    def monthly_expenses(self):
        return self._currency_value(200, 15000)

    def account_balance(self):
        return self._currency_value(0, 200000)

    def loan_amount(self):
        return self._currency_value(500, 500000)

    def annual_bonus(self):
        return self._currency_value(0, 100000)

    def credit_limit(self):
        return self._currency_value(1000, 80000)

    def social_security_payment(self):
        return self._currency_value(200, 4000)

    def tax_payment(self):
        return self._currency_value(100, 50000)

    def debt_ratio(self):
        pct = round(random.uniform(0.01, 1.50), 2)  # 1%–150%
        return f"{pct}%"

    def investment_return(self):
        pct = round(random.uniform(-0.50, 1.00), 2)  # -50% to 100%
        return f"{pct}%"

    def roi(self):
        pct = round(random.uniform(-0.20, 0.60), 2)  # ROI -20% to 60%
        return f"{pct}%"

    def credit_score(self):
        return random.randint(300, 850)

    # ---- RANDOM GENERATOR ----
    def generate_random(self):
        methods = [
            self.monthly_income,
            self.monthly_expenses,
            self.account_balance,
            self.loan_amount,
            self.annual_bonus,
            self.credit_limit,
            self.social_security_payment,
            self.tax_payment,
            self.debt_ratio,
            self.investment_return,
            self.roi,
            self.credit_score
        ]
        method = random.choice([m for m in methods if m.__name__ != self.last_method])
        self.last_method = method.__name__
        return " ".join(method.__name__.split("_")), method()
    
class Education:

    def __init__(self, context):
        self.context = context
        self.last_method = None

    def academic_degree(self):
        base_prompt = f"Generate academic degree of the person."
        base_prompt += f"It should make sense, based on this context: {self.context}"
        base_prompt += f"Output only the value, no explanations. Keep it between 1-3 words."
        return call_openai(base_prompt)

    def education_level(self):
        base_prompt = f"Generate education level of the person."
        base_prompt += f"It should make sense, based on this context: {self.context}"
        base_prompt += f"Output only the value, no explanations. Keep it between 1-3 words."
        return call_openai(base_prompt)

    def generate_random(self):
        methods = [self.academic_degree, self.education_level]
        method = random.choice([m for m in methods if m.__name__ != self.last_method])
        self.last_method = method.__name__
        return " ".join(method.__name__.split("_")), method()

class Appearance:

    def __init__(self, context):
        self.context = context
        self.last_method = None

    def height(self):
        # 150–200 cm or 5'0"–6'6"
        if random.random() < 0.5:
            return f"{random.randint(150,200)}cm"
        else:
            return f"{random.randint(5,6)}’{random.randint(0,11)}\""

    def weight(self):
        return f"{random.randint(40,130)}kg"

    def blood_type(self):
        return random.choice(["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])

    def physical_features(self):
        base_prompt = (
        "Generate a realistic physical feature a person might have on their face or body. "
        "Avoid tattoos or tattoo-related outputs unless strongly justified. "
        "Examples include scars, freckles, birthmarks, dimples, piercings, glasses, etc. "
        f"Use the following context only for realism: {self.context}\n"
        "Output only the feature, 1–3 words, no explanations."
    )
        return call_openai(base_prompt)

    def generate_random(self):
        methods = [
            self.height,
            self.weight,
            self.blood_type,
            self.physical_features
        ]
        method = random.choice([m for m in methods if m.__name__ != self.last_method])
        self.last_method = method.__name__
        return " ".join(method.__name__.split("_")), method()
    
class Health:

    def __init__(self, context):
        self.context = context
        self.last_method = None

    def disability_status(self):
        base_prompt = f"""
        Generate a realistic disability status for a person.
        Include a mix of conditions: physical, sensory, cognitive, mental health, and chronic illness.
        Do not always return "no disability"—use it only occasionally.
        Examples include:
        - hearing impairment
        - mobility impairment
        - vision loss
        - ADHD
        - PTSD
        - autism spectrum
        - chronic pain
        - wheelchair user
        - partial blindness
        - no disability

        Context (may influence realism but should not force any outcome):
        {self.context}

        Output only the status, 1–3 words, no explanations.
        """
        return call_openai(base_prompt)

    def medical_condition(self):
        base_prompt = f"Generate medical condition."
        base_prompt += f"It should make sense, based on this context: {self.context}"
        base_prompt += f"Output only the value, no explanations. Keep it between 1-3 words."
        return call_openai(base_prompt)

    def generate_random(self):
        methods = [self.disability_status, self.medical_condition]
        method = random.choice([m for m in methods if m.__name__ != self.last_method])
        self.last_method = method.__name__
        return " ".join(method.__name__.split("_")), method()

class Demographic:

    def __init__(self, context):
        self.context = context
        self.last_method = None

    def nationality(self):
        base_prompt = f"""
        Generate a real nationality (country-based identity), not an organization or group.
        Do not overuse common nationalities like American, Canadian, or British—use them only sometimes.
        Use only actual nationalities from real countries.
        Examples:
        - Mexican
        - Brazilian
        - British
        - Ukrainian

        Do not generate fictional names, political groups, or organizations.
        Context (optional for realism):
        {self.context}

        Output only the nationality, 1–2 words, no explanations.
        """
        return call_openai(base_prompt)

    def gender(self):
        return random.choice(["Male", "Female", "Non-binary"]).lower()      

    def ethnicity(self):
        base_prompt = f"""
        Generate a realistic ethnicity category someone might identify with.
        Use widely recognized ethnicity classifications (not umbrella or ambiguous labels).
        Avoid vague terms like multiracial, biracial, mixed, or generic descriptors.
        Examples include:
        - Hispanic / Latino
        - African American
        - Arab
        - South Asian

        Context for realism only (do not assume mixed or undefined based on context):
        {self.context}

        Output only the ethnicity, 1–2 words, no explanations.
        """
        return call_openai(base_prompt)

    def race(self):
        base_prompt = f"""
        Generate a realistic human race.
        Choose from globally recognized racial categories only.
        Examples include:
        - White
        - Black or African
        - Asian
        - Indigenous or Native American
        - Pacific Islander
        - Middle Eastern
        - Mixed Race

        Do not generate abstract concepts, organizations, or identity groups.
        Context (optional for realism, but do not infer fictional group names):
        {self.context}

        Output only one race label, 1 word. No explanations.
        """
        return call_openai(base_prompt)

    def generate_random(self):
        methods = [self.nationality, self.ethnicity, self.race, self.gender]
        method = random.choice([m for m in methods if m.__name__ != self.last_method])
        self.last_method = method.__name__
        return " ".join(method.__name__.split("_")), method()

class Belief:

    def __init__(self, context):
        self.context = context
        self.last_method = None

    def religious_belief(self):
        beliefs = [
        "Christianity",
        "Islam",
        "Judaism",
        "Hinduism",
        "Buddhism",
        "Atheist",
        "Agnostic",
        "Spiritual",
        "Taoism", "Catholic"
    ]
        return random.choice(beliefs)

    def political_affiliation(self):
        affiliations = [
        "Liberal",
        "Conservative",
        "Independent",
        "Progressive",
        "Green",
        "Libertarian",
        "Socialist",
        "Centrist",
        "Nationalist",
        "Social Democrat"
    ]

        return random.choice(affiliations)

    def generate_random(self):
        methods = [self.religious_belief, self.political_affiliation]
        method = random.choice([m for m in methods if m.__name__ != self.last_method])
        self.last_method = method.__name__
        return " ".join(method.__name__.split("_")), method()
    
class Organization:

    def __init__(self, context):
        self.context = context + f" Country: {fake.country()}"
        self.last_method = None

    def company_name(self):
        base_prompt = f"Generate a name of the real company."
        base_prompt += f"It should make sense, based on this context: {self.context}"
        base_prompt += f"Output only the value, no explanations. Keep it between 1-3 words."
        return call_openai(base_prompt)

    def educational_institution(self):
        base_prompt = f"Generate a name of the real education institution."
        base_prompt += f"It should make sense, based on this context: {self.context}"
        base_prompt += f"Output only the value, no explanations. Keep it between 1-3 words."
        return call_openai(base_prompt)

    def government_agency(self):
        base_prompt = f"Generate a name of the real governmental agency."
        base_prompt += f"It should make sense, based on this context: {self.context}"
        base_prompt += f"Output only the value, no explanations. Keep it between 1-3 words."
        return call_openai(base_prompt)

    def ngo(self):
        base_prompt = f"Generate a name of the real non-governmental organization."
        base_prompt += f"It should make sense, based on this context: {self.context}"
        base_prompt += f"Output only the value, no explanations. Keep it between 1-3 words."
        return call_openai(base_prompt)

    def healthcare_facility(self):
        base_prompt = f"Generate a name of the real healthcare facility."
        base_prompt += f"It should make sense, based on this context: {self.context}"
        base_prompt += f"Output only the value, no explanations. Keep it between 1-3 words."
        return call_openai(base_prompt)

    # ------ RANDOM GENERATOR ------
    def generate_random(self):
        methods = [
            self.company_name,
            self.educational_institution,
            self.government_agency,
            self.ngo,
            self.healthcare_facility
        ]
        method = random.choice([m for m in methods if m.__name__ != self.last_method])
        self.last_method = method.__name__
        return " ".join(method.__name__.split("_")), method()

class Location:

    def __init__(self, context):
        self.context = context + f" Country: {fake.country()}"
        self.last_method = None

    def street_address(self):
        # synthetic style formatting
        base_prompt = f"Generate a realistic street address."
        base_prompt += f"It should make sense, based on this context: {self.context}"
        base_prompt += f"Output only the value, no exaplantions."
        return call_openai(base_prompt)

    def city_region(self):
        base_prompt = f"Generate a city name or region."
        base_prompt += f"It should make sense, based on this context: {self.context}"
        base_prompt += f"Output only the value, no exaplantions."
        return call_openai(base_prompt)

    def landmark(self):
        base_prompt = f"Generate a name of the real work landmark."
        base_prompt += f"It should make sense, based on this context: {self.context}"
        base_prompt += f"Output only the value, no exaplantions."
        return call_openai(base_prompt)

    # ------- RANDOM SELECTOR -------
    def generate_random(self):
        methods = [
            self.street_address,
            self.city_region,
            self.landmark
        ]
        method = random.choice([m for m in methods if m.__name__ != self.last_method])
        self.last_method = method.__name__
        return " ".join(method.__name__.split("_")), method()
    
class Age:
    def __init__(self, context):
        self.context = context
        self.last_method = None

    def age(self):
        base_prompt = f"Generate a realistic age."
        base_prompt += f"It should make sense, based on this context: {self.context}"
        base_prompt += f"Output only the value, no exaplantions."
        return call_openai(base_prompt)

    def generate_random(self):
        return "age", self.age()

class Occupation:
    def __init__(self, context):
        self.context = context
        self.last_method = None

    def occupation(self):
        base_prompt = f"""
        Generate a realistic occupation or profession.
        Include diverse categories such as service work, trades, medical, education,
        transportation, creative fields, and technology.
        Avoid always returning roles like analyst, specialist, consultant, or officer —
        they are allowed sometimes, but variety is important.

        Examples:
        teacher, carpenter, paramedic, chef, software engineer,
        driver, artist, nurse, electrician, journalist, farmer.

        Context for inspiration: {self.context}

        Output only the job title, 1–3 words. Do not explain.
        """
        return call_openai(base_prompt).lower()

    def generate_random(self):
        return "occupation", self.occupation()


class Name:

    def __init__(self, context):
        self.context = context
        self.last_method = None

    def first_name(self):
        first = ["Alex", "Maria", "John", "Sofia", "David", "Olena", "James", "Artem", "Emily", "Liam"]
        return random.choice(first)

    def last_name(self):
        last = ["Smith", "Brown", "Johnson", "Williams", "Miller", "Taylor", "Ponomarenko", "Chen", "Singh", "Kovalenko"]
        return random.choice(last)

    def full_name(self):
        return f"{self.first_name()} {self.last_name()}"

    def generate_random(self):
        methods = [self.first_name, self.last_name, self.full_name]
        method = random.choice([m for m in methods if m.__name__ != self.last_method])
        self.last_method = method.__name__
        return " ".join(method.__name__.split("_")), method()
    
class Relationship(CategoryBase):

    def __init__(self, context):
        self.context = context
        self.last_method = None
    def family_member(self):
        members = [
        "mother", "father",
        "sister", "brother",
        "wife", "husband",
        "son", "daughter",
        "grandmother", "grandfather",
        "uncle", "aunt",
        "cousin", "nephew", "niece"
    ]
        return random.choice(members)

    def girlfriend(self):
        gfs = ["girlfriend", "fiancée", "gf"]
        return random.choice(gfs)

    def boyfriend(self):
        bfs = ["boyfriend", "fiancé", "bf"]
        return random.choice(bfs)

    def relationship_status(self):
        statuses = [
            "single", "in a relationship", "engaged",
            "married", "divorced", "widowed",
            "separated"
        ]
        return random.choice(statuses)

    def generate_random(self):
        methods = [
            self.family_member,
            self.girlfriend,
            self.boyfriend,
            self.relationship_status
        ]
        method = random.choice([m for m in methods if m.__name__ != self.last_method])
        self.last_method = method.__name__
        return " ".join(method.__name__.split("_")), method()
    
class SexualOrientation(CategoryBase):

    def __init__(self, context):
        self.context = context
        self.last_method = None

    def sexual_orientation(self):
        statuses = [
            "homosexual", "heterosexual", "lesbian",
            "gay", "bisexual", "asexual"
        ]
        return random.choice(statuses)

    def generate_random(self):
        return "sexual orientation", self.sexual_orientation()

class PIIGenerator:

    def __init__(self, context=None):
        self.name_generator = Name(context)
        self.age_generator = Age(context)
        self.occupation_generator = Occupation(context)

        self.code_generator = Code(context)
        self.datetime_generator = Datetime(context)
        self.finance_generator = Finance(context)

        self.education_generator = Education(context)
        self.appearance_generator = Appearance(context)
        self.health_generator = Health(context)
        self.demographic_generator = Demographic(context)
        self.belief_generator = Belief(context)

        self.organization_generator = Organization(context)
        self.location_generator = Location(context)
        self.relationship_generator = Relationship(context)
        self.sexual_orientation = SexualOrientation(context)

        self.last_generator = None

        self.generators = [
            self.name_generator,
            self.age_generator,
            self.occupation_generator,
            self.code_generator,
            self.datetime_generator,
            self.finance_generator,
            self.education_generator,
            self.appearance_generator,
            self.health_generator,
            self.demographic_generator,
            self.belief_generator,
            self.organization_generator,
            self.location_generator,
            self.relationship_generator,
            self.sexual_orientation
        ]

    def generate_random(self, category=None):
        generator = None
        if category is None:
            generator = random.choice([m for m in self.generators if m.__class__.__name__ != self.last_generator])
            self.last_generator = generator.__class__.__name__
            field_name, value = generator.generate_random()
            category_name = generator.__class__.__name__
        else:
            if category == "demographic":
                generator = self.demographic_generator
            elif category == "age":
                generator = self.age_generator
            elif category == "occupation":
                generator = self.occupation_generator
            elif category == "education":
                generator = self.education_generator
            elif category == "location":
                generator = self.location_generator
            elif category == "organization":
                generator = self.organization_generator
            elif category == "health":
                generator = self.health_generator
            elif category == "sexual orientation":
                generator = self.sexual_orientation
            elif category == "family":
                generator = self.relationship_generator
            elif category == "finance":
                generator= self.finance_generator
            elif category == "code":
                generator= self.code_generator
            elif category == "name":
                generator= self.name_generator
            elif category == "datetime":
                generator= self.datetime_generator
            elif category == "belief":
                generator= self.belief_generator
            elif category == "appearance":
                generator= self.appearance_generator
  
            field_name, value = generator.generate_random()
            category_name = generator.__class__.__name__
        return category_name.lower(), field_name.lower(), value
    
