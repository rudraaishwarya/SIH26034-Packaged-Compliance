import os
import re
import tempfile

import streamlit as st
from paddleocr import PaddleOCR


# ============================================================
# PADDLEOCR CONFIGURATION
# ============================================================

os.environ["FLAGS_enable_pir_api"] = "0"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PackSure",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# LOAD OCR
# ============================================================

@st.cache_resource
def load_ocr():
    return PaddleOCR(
        lang="en",
        enable_mkldnn=False
    )


ocr = load_ocr()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text):
    """Basic OCR text cleaning."""
    if not text:
        return ""

    text = str(text)

    text = text.replace("\\", "")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_for_matching(text):
    """Create normalized text for keyword matching."""
    text = clean_text(text).upper()

    text = text.replace("₹", " RS ")
    text = text.replace(":", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_internal_marker(text):
    """Reject our own internal markers."""
    text = clean_text(text)

    patterns = [
        r"^---\s*PHOTO\s*\d+\s*---$",
        r"^PHOTO\s*\d+$",
        r"^IMAGE\s*\d+$"
    ]

    return any(re.match(pattern, text, re.I) for pattern in patterns)


# ============================================================
# DATE EXTRACTION
# ============================================================

def normalize_date_candidate(text):
    """
    Normalize OCR noise around dates.

    Examples:
    #05/26       -> 05/26
    @04/29(M1)   -> 04/29
    $07/26       -> 07/26
    ￥12/29      -> 12/29
    """

    if not text:
        return ""

    text = str(text)

    # Remove OCR symbols before dates
    text = re.sub(r"[#@$₹¥￥*]+", " ", text)

    # Remove brackets but keep content
    text = re.sub(r"[\(\)\[\]\{\}]", " ", text)

    return text


def repair_spaced_dates(text):
    """
    Repair OCR cases like:

    16/0 5/20 2 6
    15/1 1 /20 27

    into candidate strings that can be detected.
    """

    if not text:
        return ""

    # Only normalize spacing around slash-separated numbers
    text = re.sub(r"(\d)\s*/\s*(\d)", r"\1/\2", text)

    # Remove spaces BETWEEN digits only
    # This helps OCR fragments such as 0 5 -> 05
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)

    return text


def extract_dates_from_text(text):
    """
    Extract valid package-style dates.

    Supports:
    DD/MM/YY
    DD/MM/YYYY
    DD-MM-YY
    DD-MM-YYYY
    MM/YY
    MM/YYYY
    """

    if not text:
        return []

    original = normalize_date_candidate(text)

    versions = [
        original,
        repair_spaced_dates(original)
    ]

    found = []

    patterns = [

        # DD/MM/YYYY or DD/MM/YY
        r"(?<!\d)(0?[1-9]|[12]\d|3[01])\s*[/-]\s*(0?[1-9]|1[0-2])\s*[/-]\s*(\d{2,4})(?!\d)",

        # MM/YYYY or MM/YY
        r"(?<!\d)(0?[1-9]|1[0-2])\s*[/-]\s*(\d{2,4})(?!\d)"
    ]

    for version in versions:

        for pattern in patterns:

            for match in re.finditer(pattern, version):

                value = match.group(0)

                # Remove spaces around separators
                value = re.sub(r"\s+", "", value)

                # Reject obvious telephone-like fragments
                if len(re.sub(r"\D", "", value)) < 4:
                    continue

                # Validate full day/month dates
                parts = re.split(r"[/-]", value)

                try:

                    if len(parts) == 3:

                        day = int(parts[0])
                        month = int(parts[1])

                        if not (1 <= day <= 31 and 1 <= month <= 12):
                            continue

                    elif len(parts) == 2:

                        month = int(parts[0])

                        if not (1 <= month <= 12):
                            continue

                except ValueError:
                    continue

                if value not in found:
                    found.append(value)

    # Prefer longer complete dates
    found.sort(key=lambda x: (-len(x), x))

    return found


def extract_dates_in_order(lines):
    """
    Extract dates while preserving OCR row order.

    Important for:
    Manufacturing Date -> first date
    Expiry Date       -> second date
    """

    ordered_dates = []

    for line_index, line in enumerate(lines):

        dates = extract_dates_from_text(line)

        for date in dates:

            item = {
                "date": date,
                "line_index": line_index,
                "line": line
            }

            # Avoid duplicates
            if not any(
                existing["date"] == date
                for existing in ordered_dates
            ):
                ordered_dates.append(item)

    # Keep OCR order
    ordered_dates.sort(key=lambda x: x["line_index"])

    return ordered_dates


def is_date_label(line, keywords):
    normalized = normalize_for_matching(line)

    return any(keyword in normalized for keyword in keywords)


def extract_manufacturing_and_expiry(lines):
    """
    Priority:

    1. Strong label association
    2. Nearby row association
    3. Two-date fallback
    """

    manufacturing_keywords = [
        "MFD",
        "MFG",
        "MFG DATE",
        "MFD DATE",
        "MANUFACTURED",
        "MANUFACTURING DATE",
        "DATE OF MFG",
        "DATE OF MANUFACTURE",
        "PACKED ON",
        "PACKING DATE",
        "PKD"
    ]

    expiry_keywords = [
        "EXP",
        "EXPIRY",
        "EXP DATE",
        "EXPIRY DATE",
        "USE BY",
        "USE BEFORE",
        "BEST BEFORE"
    ]

    manufacturing_date = None
    expiry_date = None

    # --------------------------------------------------------
    # STEP 1: Label + same row / nearby rows
    # --------------------------------------------------------

    for i, line in enumerate(lines):

        normalized = normalize_for_matching(line)

        # ---------------- MFD ----------------

        if manufacturing_date is None and is_date_label(
            line,
            manufacturing_keywords
        ):

            # Same row
            dates = extract_dates_from_text(line)

            if dates:
                manufacturing_date = dates[0]

            # Next 3 rows
            if manufacturing_date is None:

                for j in range(i + 1, min(i + 4, len(lines))):

                    dates = extract_dates_from_text(lines[j])

                    if dates:
                        manufacturing_date = dates[0]
                        break

        # ---------------- EXPIRY ----------------

        if expiry_date is None and is_date_label(
            line,
            expiry_keywords
        ):

            dates = extract_dates_from_text(line)

            if dates:
                expiry_date = dates[0]

            # Next 3 rows
            if expiry_date is None:

                for j in range(i + 1, min(i + 4, len(lines))):

                    dates = extract_dates_from_text(lines[j])

                    if dates:
                        expiry_date = dates[0]
                        break

    # --------------------------------------------------------
    # STEP 2: TWO-DATE FALLBACK
    # --------------------------------------------------------

    ordered_dates = extract_dates_in_order(lines)

    # Remove dates that appear in obvious phone/contact rows
    filtered_dates = []

    phone_keywords = [
        "TOLL",
        "PHONE",
        "CONTACT",
        "CALL",
        "HELPLINE"
    ]

    for item in ordered_dates:

        normalized_line = normalize_for_matching(item["line"])

        if any(keyword in normalized_line for keyword in phone_keywords):
            continue

        filtered_dates.append(item)

    # If we have dates but labels failed:
    # first plausible date = manufacturing
    # second plausible date = expiry

    if manufacturing_date is None and len(filtered_dates) >= 1:
        manufacturing_date = filtered_dates[0]["date"]

    if expiry_date is None and len(filtered_dates) >= 2:

        # Don't duplicate manufacturing date
        for item in filtered_dates:

            if item["date"] != manufacturing_date:
                expiry_date = item["date"]
                break

    # If manufacturing and expiry accidentally same
    if manufacturing_date == expiry_date:
        expiry_date = None

    return manufacturing_date, expiry_date


# ============================================================
# NET QUANTITY EXTRACTION
# ============================================================

def format_quantity(number, unit):

    number = number.strip()
    unit = unit.lower()

    if unit == "kgs":
        unit = "kg"

    if unit == "grams":
        unit = "g"

    if unit == "litres":
        unit = "l"

    if unit == "liters":
        unit = "l"

    return f"{number} {unit}"


def extract_net_quantity(lines):

    # Keywords strongly indicating quantity
    strong_keywords = [
        "NET WEIGHT",
        "NET WT",
        "NET QUANTITY",
        "NET CONTENT",
        "NET VOLUME",
        "NET VOL",
        "NETWEIGHT"
    ]

    quantity_pattern = re.compile(
        r"(?<![\w.])"
        r"(\d+(?:\.\d+)?)"
        r"\s*"
        r"(kg|kgs|g|grams|ml|l|litre|litres|liter|liters)"
        r"(?!\s*/)",
        re.I
    )

    candidates = []

    for i, line in enumerate(lines):

        normalized = normalize_for_matching(line)

        # Reject price-per-unit lines
        if re.search(
            r"(RS|₹)\s*\d+(?:\.\d+)?\s*/\s*(G|KG|ML|L)",
            normalized
        ):
            continue

        # ----------------------------------------------------
        # STRONG KEYWORD IN SAME LINE
        # ----------------------------------------------------

        if any(keyword in normalized for keyword in strong_keywords):

            matches = quantity_pattern.findall(line)

            for number, unit in matches:

                candidates.append({
                    "value": format_quantity(number, unit),
                    "score": 100
                })

        # ----------------------------------------------------
        # QUANTITY FOLLOWED BY "NET"
        # Example: 550ml net
        # ----------------------------------------------------

        pattern = re.search(
            r"(\d+(?:\.\d+)?)\s*(kg|g|ml|l)\s*NET\b",
            normalized,
            re.I
        )

        if pattern:

            candidates.append({
                "value": format_quantity(
                    pattern.group(1),
                    pattern.group(2)
                ),
                "score": 90
            })

        # ----------------------------------------------------
        # Look at nearby rows after NET WEIGHT heading
        # ----------------------------------------------------

        if any(keyword in normalized for keyword in strong_keywords):

            for j in range(i + 1, min(i + 3, len(lines))):

                nearby = lines[j]

                matches = quantity_pattern.findall(nearby)

                for number, unit in matches:

                    candidates.append({
                        "value": format_quantity(number, unit),
                        "score": 80
                    })

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["score"], reverse=True)

    return candidates[0]["value"]


# ============================================================
# MANUFACTURER EXTRACTION
# ============================================================

def clean_company_text(text):

    text = clean_text(text)

    # Remove keyword portion
    text = re.sub(
        r"(?i).*?(MANUFACTURED\s+BY|MANUFACTURED\s*&\s*MARKETED\s+BY|MANUFACTURER|MFD\.?\s*BY|PACKED\s+BY|MARKETED\s+BY)\s*:?\s*",
        "",
        text
    )

    # Stop at unrelated sections
    stop_words = [
        "CONSUMER CARE",
        "CUSTOMER CARE",
        "COMPLAINT",
        "CONTACT",
        "TOLL FREE",
        "EMAIL",
        "E-MAIL",
        "FSSAI",
        "MRP",
        "BATCH",
        "NET WEIGHT",
        "NET VOLUME"
    ]

    upper = text.upper()

    cut_positions = []

    for word in stop_words:

        pos = upper.find(word)

        if pos != -1:
            cut_positions.append(pos)

    if cut_positions:
        text = text[:min(cut_positions)]

    text = text.strip(" :-|,.")

    return text


def looks_like_company(text):

    normalized = normalize_for_matching(text)

    company_indicators = [
        "PVT",
        "PRIVATE",
        "LTD",
        "LIMITED",
        "LLP",
        "INDUSTRIES",
        "FOODS",
        "UNILEVER",
        "ITC"
    ]

    return any(
        indicator in normalized
        for indicator in company_indicators
    )


def extract_manufacturer(lines):

    keywords = [
        "MANUFACTURED BY",
        "MANUFACTURED & MARKETED BY",
        "MANUFACTURER",
        "MFD BY",
        "PACKED BY",
        "MARKETED BY"
    ]

    stop_keywords = [
        "CONSUMER CARE",
        "CUSTOMER CARE",
        "COMPLAINT",
        "CONTACT",
        "TOLL FREE",
        "EMAIL"
    ]

    candidates = []

    for i, line in enumerate(lines):

        normalized = normalize_for_matching(line)

        if any(keyword in normalized for keyword in keywords):

            # Same line
            cleaned = clean_company_text(line)

            if len(cleaned) >= 3:

                candidates.append(cleaned)

            # Check only immediate next rows
            # and stop quickly to avoid huge paragraphs
            for j in range(i + 1, min(i + 3, len(lines))):

                next_line = clean_text(lines[j])

                next_normalized = normalize_for_matching(next_line)

                if any(
                    keyword in next_normalized
                    for keyword in stop_keywords
                ):
                    break

                if looks_like_company(next_line):

                    candidates.append(
                        clean_company_text(next_line)
                    )
                    break

    # Remove bad/long candidates
    valid = []

    for candidate in candidates:

        candidate = clean_company_text(candidate)

        if len(candidate) < 3:
            continue

        # Don't allow huge OCR paragraphs
        if len(candidate) > 150:
            continue

        if candidate not in valid:
            valid.append(candidate)

    if not valid:
        return None

    # Prefer company-looking text
    for candidate in valid:

        if looks_like_company(candidate):
            return candidate

    return valid[0]


# ============================================================
# MRP EXTRACTION
# ============================================================

def extract_mrp(lines):

    patterns = [

        r"(?:MRP|M\.R\.P\.?)\s*[:\-]?\s*(?:RS\.?|₹)?\s*(\d+(?:\.\d{1,2})?)",

        r"(?:RS\.?|₹)\s*(\d+(?:\.\d{1,2})?)"
    ]

    # First pass: lines containing MRP
    for line in lines:

        normalized = normalize_for_matching(line)

        if "MRP" in normalized:

            for pattern in patterns:

                match = re.search(pattern, line, re.I)

                if match:

                    try:
                        value = float(match.group(1))

                        # Ignore absurd OCR values
                        if 0 < value < 100000:
                            return f"₹{value:.2f}"

                    except ValueError:
                        pass

    # Second pass: Rs.250 style
    for line in lines:

        if re.search(r"(RS\.?|₹)\s*\d+", line, re.I):

            match = re.search(
                r"(?:RS\.?|₹)\s*(\d+(?:\.\d{1,2})?)",
                line,
                re.I
            )

            if match:

                try:

                    value = float(match.group(1))

                    if 0 < value < 100000:
                        return f"₹{value:.2f}"

                except ValueError:
                    pass

    return None


# ============================================================
# BATCH NUMBER EXTRACTION
# ============================================================

def extract_batch_number(lines):

    batch_keywords = [
        "BATCH NO",
        "BATCHNO",
        "BATCH NUMBER",
        "B. NO",
        "LOT NO",
        "LOT NUMBER"
    ]

    invalid_values = {
        "PKD",
        "MFD",
        "MFG",
        "EXP",
        "DATE",
        "NET",
        "MRP",
        "SECTOR",
        "SECTOR-"
    }

    for i, line in enumerate(lines):

        normalized = normalize_for_matching(line)

        if any(keyword in normalized for keyword in batch_keywords):

            # Remove label
            value = re.sub(
                r"(?i).*?(BATCH\s*NO\.?|BATCHNO\.?|BATCH\s*NUMBER|B\.\s*NO\.?|LOT\s*NO\.?|LOT\s*NUMBER)\s*[:\-]?\s*",
                "",
                line
            )

            value = clean_text(value)

            # Take first meaningful code
            match = re.search(
                r"\b[A-Z0-9][A-Z0-9\-\/]{2,20}\b",
                value,
                re.I
            )

            if match:

                candidate = match.group(0)

                if candidate.upper() not in invalid_values:
                    return candidate

            # Nearby row fallback
            if i + 1 < len(lines):

                next_line = clean_text(lines[i + 1])

                match = re.search(
                    r"\b[A-Z0-9][A-Z0-9\-\/]{2,20}\b",
                    next_line,
                    re.I
                )

                if match:

                    candidate = match.group(0)

                    if candidate.upper() not in invalid_values:
                        return candidate

    return None


# ============================================================
# FSSAI EXTRACTION
# ============================================================

def extract_fssai(lines):

    # Standard Indian FSSAI licence number is 14 digits
    for line in lines:

        numbers = re.findall(r"(?<!\d)(\d{14})(?!\d)", line)

        if numbers:
            return numbers[0]

    # Combined OCR fallback
    combined = " ".join(lines)

    numbers = re.findall(r"(?<!\d)(\d{14})(?!\d)", combined)

    if numbers:
        return numbers[0]

    return None


# ============================================================
# PRODUCT NAME EXTRACTION
# ============================================================

def extract_product_name(lines):

    ignored_keywords = [
        "INGREDIENT",
        "NUTRITION",
        "MARKETED BY",
        "MANUFACTURED BY",
        "NET WEIGHT",
        "NET VOLUME",
        "FSSAI",
        "MRP",
        "BATCH",
        "MFD",
        "MFG",
        "EXP",
        "USE BY",
        "BEST BEFORE",
        "CONSUMER",
        "CONTACT",
        "TOLL FREE",
        "EMAIL"
    ]

    for line in lines[:15]:

        line = clean_text(line)

        if not line:
            continue

        if is_internal_marker(line):
            continue

        normalized = normalize_for_matching(line)

        if any(keyword in normalized for keyword in ignored_keywords):
            continue

        # Avoid sentences
        if len(line) > 60:
            continue

        # Prefer alphabetic product/brand-like lines
        letters = sum(char.isalpha() for char in line)

        if letters >= 3:
            return line

    return None


# ============================================================
# EXTRACT ALL PACKAGE INFORMATION
# ============================================================

def extract_package_information(all_lines):

    # Remove empty lines and internal markers
    lines = [
        clean_text(line)
        for line in all_lines
        if clean_text(line)
        and not is_internal_marker(line)
    ]

    manufacturing_date, expiry_date = (
        extract_manufacturing_and_expiry(lines)
    )

    return {

        "product_name": extract_product_name(lines),

        "net_quantity": extract_net_quantity(lines),

        "manufacturer": extract_manufacturer(lines),

        "manufacturing_date": manufacturing_date,

        "expiry_date": expiry_date,

        "mrp": extract_mrp(lines),

        "batch_number": extract_batch_number(lines),

        "fssai": extract_fssai(lines)
    }


# ============================================================
# OCR PROCESSING
# ============================================================

def process_uploaded_image(uploaded_file):

    temp_path = None

    try:

        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())

            temp_path = temp_file.name

        # IMPORTANT:
        # Compatible with your currently working PaddleOCR setup
        result = ocr.predict(temp_path)

        texts = []

        for res in result:

            data = res.json

            if not isinstance(data, dict):
                continue

            res_data = data.get("res", {})

            recognized_texts = res_data.get(
                "rec_texts",
                []
            )

            for text in recognized_texts:

                text = clean_text(text)

                if text:
                    texts.append(text)

        return texts

    except Exception as e:

        st.error(
            f"❌ OCR error while processing "
            f"{uploaded_file.name}: {e}"
        )

        return []

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# UI HELPER
# ============================================================

def display_priority_field(title, icon, value):

    st.markdown(f"### {icon} {title}")

    if value:

        st.success(value)

    else:

        st.warning("Not detected — Needs Review")


# ============================================================
# MAIN UI
# ============================================================

st.title("📦 PackSure")

st.caption(
    "AI-assisted packaged commodity information extraction "
    "and compliance analysis"
)

st.write(
    "Upload multiple photos of the **same packaged product** "
    "(front, back, side, bottom, etc.). "
    "PackSure combines information from all images to analyse "
    "**one product package**."
)


uploaded_files = st.file_uploader(
    "📷 Upload photos of ONE product package",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# ============================================================
# PROCESS IMAGES
# ============================================================

if uploaded_files:

    st.markdown(
        f"### 📦 {len(uploaded_files)} package photo(s) uploaded"
    )

    st.markdown("---")

    st.markdown("### 📷 Uploaded Package Images")

    all_ocr_lines = []

    # --------------------------------------------------------
    # DISPLAY IMAGES + OCR
    # --------------------------------------------------------

    for index, uploaded_file in enumerate(uploaded_files):

        st.markdown(
            f"#### 📷 Product Photo {index + 1}"
        )

        st.image(
            uploaded_file,
            width="stretch"
        )

        with st.spinner(
            f"Extracting text from Photo {index + 1}..."
        ):

            image_lines = process_uploaded_image(
                uploaded_file
            )

        all_ocr_lines.extend(image_lines)

        with st.expander(
            f"📝 View OCR Text from Photo {index + 1}"
        ):

            if image_lines:

                st.text_area(
                    f"OCR Result {index + 1}",
                    "\n".join(image_lines),
                    height=280,
                    key=f"ocr_{index}"
                )

            else:

                st.warning(
                    "⚠️ No text detected in this image."
                )

        st.markdown("---")

    # ========================================================
    # COMBINED ANALYSIS
    # ========================================================

    package_info = extract_package_information(
        all_ocr_lines
    )

    st.markdown("## 📦 PackSure Package Analysis")

    st.markdown(
        "### 🎯 Priority Compliance Information"
    )

    # --------------------------------------------------------
    # PRIORITY CARDS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        display_priority_field(
            "Net Quantity",
            "⚖️",
            package_info["net_quantity"]
        )

    with col2:

        display_priority_field(
            "Manufacturer / Packer",
            "🏭",
            package_info["manufacturer"]
        )

    col3, col4 = st.columns(2)

    with col3:

        display_priority_field(
            "Manufacturing / Packing Date",
            "📅",
            package_info["manufacturing_date"]
        )

    with col4:

        display_priority_field(
            "Expiry / Use By Date",
            "⏳",
            package_info["expiry_date"]
        )

    # ========================================================
    # ADDITIONAL INFORMATION
    # ========================================================

    st.markdown("---")

    st.markdown("### 📦 Additional Package Information")

    product_name = (
        package_info["product_name"]
        or "Not detected"
    )

    mrp = (
        package_info["mrp"]
        or "Not detected"
    )

    batch = (
        package_info["batch_number"]
        or "Not detected"
    )

    fssai = (
        package_info["fssai"]
        or "Not detected"
    )

    st.markdown(
        f"**Product Name:** {product_name}"
    )

    st.markdown(
        f"**MRP:** {mrp}"
    )

    st.markdown(
        f"**Batch / Lot Number:** {batch}"
    )

    if package_info["fssai"]:

        st.markdown(
            f"🍽️ **FSSAI Licence Number:** "
            f"{fssai} ✅ Confirmed"
        )

    else:

        st.markdown(
            "🍽️ **FSSAI Licence Number:** "
            "Not detected"
        )

    # ========================================================
    # DETECTION SUMMARY
    # ========================================================

    st.markdown("---")

    st.markdown("### 📊 Detection Summary")

    priority_values = [

        package_info["net_quantity"],

        package_info["manufacturer"],

        package_info["manufacturing_date"],

        package_info["expiry_date"]
    ]

    detected_count = sum(
        value is not None
        for value in priority_values
    )

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        if package_info["net_quantity"]:
            st.success("Net Quantity detected")
        else:
            st.warning("Net Quantity needs review")

        if package_info["manufacturer"]:
            st.success("Manufacturer / Packer detected")
        else:
            st.warning("Manufacturer / Packer needs review")

    with summary_col2:

        if package_info["manufacturing_date"]:
            st.success("Manufacturing Date detected")
        else:
            st.warning("Manufacturing Date needs review")

        if package_info["expiry_date"]:
            st.success("Expiry Date detected")
        else:
            st.warning("Expiry / Use By Date needs review")

    st.info(
        f"📦 Priority fields detected: "
        f"**{detected_count}/4**"
    )

    # ========================================================
    # COMBINED OCR
    # ========================================================

    st.markdown("---")

    with st.expander(
        "📝 View Combined OCR Text from All Images"
    ):

        if all_ocr_lines:

            st.text_area(
                "Combined Package OCR Text",
                "\n".join(all_ocr_lines),
                height=400
            )

        else:

            st.warning(
                "No OCR text available."
            )

    # ========================================================
    # NEXT STEP
    # ========================================================

    st.markdown("---")

    st.info(
        "⚖️ Next step: Compare the extracted package information "
        "with compliance rules from compliance_rules.json."
    )

else:

    st.info(
        "📷 Upload one or more photos of a packaged product "
        "to begin analysis."
    )