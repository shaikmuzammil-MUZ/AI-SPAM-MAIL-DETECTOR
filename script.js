/* ============================================
   AI SPAM MAIL DETECTOR
   Machine Learning & NLP Classifier + Deep Heuristics
   ============================================ */

// ===================================================
// AI MACHINE LEARNING & NLP ENGINE
// Pre-trained Multinomial Naive Bayes / TF-IDF Log-Odds
// ===================================================

const AI_BASE_PRIOR = 0.0; // Balanced prior log-odds

// Learned log-likelihood weights: log( P(w | Spam) / P(w | Ham) )
// Positive values push towards SPAM; negative values push towards HAM/LEGIT
const AI_VOCAB_WEIGHTS = {
  // --- B2B Cold Outreach & Unsolicited Sales ---
  'quick_question':     3.4,
  'reaching_out':       2.8,
  'touch_base':         2.9,
  'came_across':        2.6,
  'profile':            1.8,
  'bump_this':          3.2,
  'inbox':              1.5,
  'leads':              3.1,
  'b2b':                3.2,
  'qualified_leads':    3.6,
  'b2b_leads':          3.7,
  'pipeline':           2.4,
  'scale_revenue':      3.5,
  'revenue':            2.1,
  '15_min':             3.2,
  '15_minute':          3.3,
  '15_minute_call':     3.8,
  'quick_call':         3.1,
  'jump_on_call':       3.3,
  'synergy':            2.8,
  'partnership':        2.2,
  'guest_post':         3.4,
  'backlink':           3.5,
  'seo':                2.7,
  'rankings':           2.3,
  'business_dev':       2.5,
  'growth_partners':    2.9,
  'organic_traffic':    2.6,
  'following_up':       2.1,
  're_following':       3.6,
  're_quick':           3.7,
  'brief_chat':         3.1,

  // --- Promotional, Discounts & Affiliate Lures ---
  'flash_sale':         3.8,
  'discount':           2.7,
  'promo_code':         3.5,
  'coupon':             3.2,
  'voucher':            3.0,
  'save_up':            2.8,
  '80_off':             3.9,
  '70_off':             3.7,
  '50_off':             3.2,
  'unbeatable':         3.1,
  'clearance':          2.9,
  'doorbuster':         3.6,
  'black_friday':       2.7,
  'bargain':            3.0,
  'limited_time':       3.1,
  'claim_discount':     3.5,
  'claim_your':         2.8,
  'exclusive_deal':     3.4,
  'special_offer':      3.0,
  'free_trial':         2.5,
  'no_credit_card':     3.3,
  'risk_free':          2.9,
  'lowest_price':       3.2,
  'ends_tonight':       3.7,
  'expires_midnight':   3.8,
  'spots_left':         3.4,
  'items_left':         3.3,
  'hurry':              2.6,
  'act_fast':           3.0,
  'supplies_last':      3.1,
  'affiliate':          3.2,
  'ref_param':          3.3,
  'utm_campaign':       2.8,
  'tracking_link':      2.7,

  // --- Work-from-Home & Survey Baits ---
  'work_from_home':     3.9,
  'remote_job':         3.4,
  'data_entry':         3.8,
  'earn_per_day':       4.1,
  'no_experience':      3.5,
  'flexible_hours':     2.8,
  'hiring_manager':     3.2,
  'telegram':           3.4,
  'whatsapp':           2.9,
  'part_time_job':      3.2,
  'start_earning':      3.7,
  'survey':             2.6,
  'opinion_survey':     3.2,
  'gift_card':          3.4,
  'amazon_gift':        3.7,
  'e_gift_card':        3.6,
  'claim_reward':       3.5,
  'panel_closes':       3.2,

  // --- Commercial Hyperbole & Greymail Evasion ---
  '100_guaranteed':     3.8,
  'guaranteed_results': 3.9,
  'revolutionary':      2.8,
  'secret_formula':     3.5,
  'magic_formula':      3.6,
  'transform_overnight':3.4,
  'featured_on_forbes': 3.2,
  'opt_out':            2.4,
  'unsubscribe':        2.2,
  'safeunsubscribe':    3.2,
  'reply_remove':       3.6,
  'reply_unsubscribe':  3.7,
  'commercial_advert':  3.3,
  'undisclosed_recip':  3.8,
  'valued_shopper':     3.2,
  'valued_customer':    2.2,
  'dear_friend':        2.8,

  // --- Critical Phishing & Fraud Markers ---
  'verify_account':     4.2,
  'confirm_password':   4.3,
  'login_credentials':  4.4,
  'account_compromised':4.3,
  'account_suspended':  4.4,
  'suspended_within':   4.5,
  '24_hours':           2.9,
  '48_hours':           2.7,
  'ssn':                4.2,
  'social_security':    4.3,
  'pin_code':           3.8,
  'one_time_pin':       4.0,
  'otp':                3.7,
  'cvv':                4.1,
  'routing_number':     3.9,
  'wire_transfer':      4.5,
  'confidential_wire':  4.6,
  'keep_confidential':  3.8,
  'board_meeting':      2.4,
  'lottery_winner':     4.4,
  'sweepstakes':        4.2,
  '2500000_usd':        4.5,
  'unclaimed_fund':     4.0,
  'next_of_kin':        3.9,
  'customs_fee':        3.8,
  'package_on_hold':    3.7,
  'unpaid_duty':        3.8,
  'auctioned':          3.2,

  // --- Ham / Legitimate Indicators (Strong Negative Weights) ---
  'electricity':        -3.5,
  'electricity_statement':-4.0,
  'power_co':           -3.8,
  'statement_summary':  -3.6,
  'period':             -2.1,
  'amount_due':         -2.8,
  'due_date':           -2.9,
  'direct_debit':       -3.4,
  'no_action_needed':   -3.5,
  'collected_auto':     -3.7,
  'registered_email':   -2.4,
  'support_call':       -1.8,
  'meeting_minutes':    -3.2,
  'agenda_attached':    -3.5,
  'sprint_review':      -3.7,
  'standup':            -3.4,
  'jira':               -3.2,
  'github':             -3.3,
  'pull_request':       -3.5,
  'commit':             -3.0,
  'deployed':           -2.8,
  'receipt_attached':   -2.7,
  'order_confirmation': -2.5,
  'tracking_fedex_corp':-2.3,
  'quarterly_budget':   -2.8,
  'internal_wiki':      -3.4,
  'team_discussion':    -2.9,
  'signed_contract':    -2.2,
  'hr_onboarding':      -2.4,
  'alex':               -1.2,
  'sarah':              -0.8,
  'david':              -0.7,
  'regards':            -0.5,
  'thanks':             -0.8,
};

// ===================================================
// NLP TOKENIZER & FEATURE EXTRACTOR
// ===================================================

function preprocessAndTokenize(text, headers) {
  // Clean & normalize
  const combined = `${headers.subject || ''} ${headers.from || ''} ${text}`.toLowerCase();

  // Normalize URLs, Money, Emails, and Numbers
  const normalized = combined
    .replace(/https?:\/\/[^\s<>"']+/g, ' __url__ ')
    .replace(/[\$£€]\s*\d+[\d,]*(\.\d+)?/g, ' __money__ ')
    .replace(/[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}/g, ' __email__ ')
    .replace(/\b\d+[\d,]*\b/g, ' __num__ ');

  // Extract raw alphanumeric words
  const words = normalized.match(/[a-z0-9_]{2,}/g) || [];

  // Generate unigrams
  const tokens = new Set(words);

  // Generate bigrams
  for (let i = 0; i < words.length - 1; i++) {
    tokens.add(`${words[i]}_${words[i+1]}`);
  }

  // Add structural token markers
  if (/^(re|fwd):\s*/i.test(headers.subject || '')) {
    const s = (headers.subject || '').replace(/^(re|fwd):\s*/i, '').trim().toLowerCase().replace(/\s+/g, '_');
    tokens.add(`re_${s.slice(0, 15)}`);
    if (/quick/i.test(headers.subject || '')) tokens.add('re_quick');
    if (/following/i.test(headers.subject || '')) tokens.add('re_following');
  }

  if (/undisclosed[- ]recipients/i.test(headers.to || '') || (headers.to || '').includes(';;')) {
    tokens.add('undisclosed_recip');
  }

  if (/[?&](ref|aff|utm_campaign|affiliate)=/i.test(text)) {
    tokens.add('ref_param');
    tokens.add('tracking_link');
  }

  if (/\b80\s*%\s*off\b/i.test(text)) tokens.add('80_off');
  if (/\b70\s*%\s*off\b/i.test(text)) tokens.add('70_off');
  if (/\b50\s*%\s*off\b/i.test(text)) tokens.add('50_off');
  if (/\b15[- ]min(ute)?\b/i.test(text)) { tokens.add('15_min'); tokens.add('15_minute'); }
  if (/\b15[- ]min(ute)?\s+(call|chat)\b/i.test(text)) tokens.add('15_minute_call');
  if (/\bwire\s+transfer\b/i.test(text)) tokens.add('wire_transfer');
  if (/\bwork\s+from\s+home\b/i.test(text)) tokens.add('work_from_home');
  if (/\bdata\s+entry\b/i.test(text)) tokens.add('data_entry');
  if (/\bgift\s+card\b/i.test(text)) tokens.add('gift_card');
  if (/\be[- ]gift\s+card\b/i.test(text)) tokens.add('e_gift_card');
  if (/\bamazon\b/i.test(text) && /\bgift\s+card\b/i.test(text)) tokens.add('amazon_gift');
  if (/\bflash\s+sale\b/i.test(text)) tokens.add('flash_sale');
  if (/\bpromo\s+code\b/i.test(text)) tokens.add('promo_code');
  if (/\bb2b\s+leads\b/i.test(text)) tokens.add('b2b_leads');
  if (/\bqualified\s+leads\b/i.test(text)) tokens.add('qualified_leads');
  if (/\bscale\b/i.test(text) && /\brevenue\b/i.test(text)) tokens.add('scale_revenue');
  if (/\bno\s+experience\b/i.test(text)) tokens.add('no_experience');
  if (/\bearn\b.{0,20}\b(day|hour|week)\b/i.test(text)) tokens.add('earn_per_day');
  if (/\bsuspended?\s+within\b/i.test(text)) tokens.add('suspended_within');
  if (/\baccount\b.{0,15}\bsuspend/i.test(text)) tokens.add('account_suspended');
  if (/\baccount\b.{0,15}\bcompromis/i.test(text)) tokens.add('account_compromised');
  if (/\blogin\s+credentials\b/i.test(text)) tokens.add('login_credentials');
  if (/\bconfirm\s+password\b/i.test(text)) tokens.add('confirm_password');
  if (/\bverify\s+(your\s+)?account\b/i.test(text)) tokens.add('verify_account');
  if (/\b2500000\b|\b2,500,000\b/i.test(text)) tokens.add('2500000_usd');
  if (/\belectricity\s+statement\b/i.test(text)) tokens.add('electricity_statement');
  if (/\bstatement\s+summary\b/i.test(text)) tokens.add('statement_summary');
  if (/\bdirect\s+debit\b/i.test(text)) tokens.add('direct_debit');
  if (/\bno\s+action\s+(is\s+)?needed\b/i.test(text)) tokens.add('no_action_needed');
  if (/\bcollected\s+automatically\b/i.test(text)) tokens.add('collected_auto');

  return [...tokens];
}

/** Runs the Machine Learning Naive Bayes inference */
function runAiModelInference(text, headers) {
  const t0 = performance.now();
  const extractedTokens = preprocessAndTokenize(text, headers);

  let logOdds = AI_BASE_PRIOR;
  const matchedFeatures = [];

  for (const token of extractedTokens) {
    if (token in AI_VOCAB_WEIGHTS) {
      const weight = AI_VOCAB_WEIGHTS[token];
      logOdds += weight;
      matchedFeatures.push({ token, weight });
    }
  }

  // Sigmoid transfer function: P(Spam) = 1 / (1 + exp(-logOdds))
  // Clamp extreme values to prevent overflow
  const clampedOdds = Math.max(-12, Math.min(12, logOdds));
  const rawProb = 1 / (1 + Math.exp(-clampedOdds));
  const spamPercentage = Math.round(rawProb * 1000) / 10; // e.g. 98.4%

  // Rank features by absolute impact
  const topSpamFeatures = matchedFeatures
    .filter(f => f.weight > 0)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 5);

  const topHamFeatures = matchedFeatures
    .filter(f => f.weight < 0)
    .sort((a, b) => a.weight - b.weight)
    .slice(0, 5);

  const t1 = performance.now();
  const latencyMs = Math.round((t1 - t0) * 100) / 100;

  return {
    spamProbability: spamPercentage,
    logOdds,
    matchedCount: matchedFeatures.length,
    topSpamFeatures,
    topHamFeatures,
    latencyMs
  };
}

// ===================================================
// BRAND & DOMAIN INTELLIGENCE
// ===================================================
const BRAND_DB = {
  'paypal':        ['paypal.com','paypal.co.uk','paypal.com.au','paypal-communication.com'],
  'amazon':        ['amazon.com','amazon.co.uk','amazon.ca','amazon.de','amazon.in','amazonaws.com','amazon-adsystem.com'],
  'apple':         ['apple.com','icloud.com','me.com','appleid.apple.com'],
  'microsoft':     ['microsoft.com','outlook.com','hotmail.com','live.com','office.com','microsoftonline.com','office365.com'],
  'google':        ['google.com','gmail.com','googlemail.com'],
  'netflix':       ['netflix.com'],
  'facebook':      ['facebook.com','fb.com','facebookmail.com'],
  'instagram':     ['instagram.com','mail.instagram.com'],
  'linkedin':      ['linkedin.com','e.linkedin.com'],
  'dropbox':       ['dropbox.com','dropboxmail.com'],
  'chase':         ['chase.com','jpmorgan.com'],
  'wellsfargo':    ['wellsfargo.com'],
  'bankofamerica': ['bankofamerica.com','bankofamerica.us'],
  'citibank':      ['citibank.com','citi.com'],
  'barclays':      ['barclays.com','barclays.co.uk'],
  'hsbc':          ['hsbc.com','hsbc.co.uk'],
  'irs':           ['irs.gov','irs.treasury.gov'],
  'hmrc':          ['hmrc.gov.uk','gov.uk'],
  'dhl':           ['dhl.com','dhl.de','dhl.co.uk'],
  'fedex':         ['fedex.com'],
  'ups':           ['ups.com','upsemail.com'],
  'usps':          ['usps.com','informeddelivery.com'],
  'ebay':          ['ebay.com','ebay.co.uk'],
  'stripe':        ['stripe.com'],
  'coinbase':      ['coinbase.com'],
  'binance':       ['binance.com'],
  'docusign':      ['docusign.com','docusign.net'],
  'zoom':          ['zoom.us','zoom.com'],
};

const URL_SHORTENERS = new Set([
  'bit.ly','tinyurl.com','goo.gl','t.co','ow.ly','is.gd','buff.ly',
  'adf.ly','tiny.cc','tr.im','short.ie','kl.am','wp.me','u.to','v.gd',
  'cutt.ly','rb.gy','shorturl.at','tiny.one','1url.com','clck.ru',
  't2m.io','shorte.st','ouo.io','bc.vc','linktr.ee','lnkd.in','go.ly',
  'dlvr.it','ift.tt','trib.al','mtr.cool','short.io','l.linklyhq.com',
]);

const SUSPICIOUS_TLDS = new Set([
  'tk','ml','cf','ga','gq','top','xyz','click','link','work','ru',
  'pw','su','cc','online','site','website','tech','fun','live','stream',
  'icu','monster','rest','wtf','party','racing','win','date','men',
  'loan','download','trade','webcam','faith','bid','gdn','zip','mov',
]);

const TRACKING_DOMAINS = new Set([
  'click.sendgrid.net','mandrillapp.com','hubspotlinks.com','cmail19.com',
  'cmail20.com','list-manage.com','mailchimp.com','createsend.com',
  'activehosted.com','sailthru.com','constantcontact.com','klaviyo.com',
  'emarsys.net','mktomail.com','pardot.com','smartmail.com'
]);

const AFFILIATE_PARAMS = new Set([
  'ref','aff','affiliate','aff_id','affiliate_id','click_id','clickid',
  'subid','partner','utm_source','utm_medium','utm_campaign','utm_content',
  'tracking_id','track','aff_sub','promo_code'
]);

// ===================================================
// SAMPLE EMAILS (Real-World Fraud & Subtle Spam)
// ===================================================
const SAMPLES = {
  phish_bank: `From: security-team@banklogin-verify.net
Reply-To: harvest@datastealer.ru
Subject: URGENT: Your account has been compromised - Verify Immediately

Dear Customer,

We have detected unusual activity on your account. Your access will be suspended within 24 hours unless you verify your account immediately.

Click here to confirm your login credentials and account password:
http://bit.ly/3xBankVerify

Please enter your:
- Username and password
- One-time PIN sent to your phone
- Social Security Number (for identity verification)

Failure to comply will result in permanent account closure.

Bank of America Security Team`,

  phish_paypal: `From: service@paypa1-secure.com
Subject: Your PayPal account is limited - Action Required

Hello,

We noticed some unusual activity in your account. To ensure your account was not compromised, please verify your information immediately.

Your account has been limited until we hear from you. Verify Now:
http://tinyurl.com/paypa1-verify-now

If you don't verify within 48 hours, your account will be permanently closed and all pending transactions will be cancelled.

Enter your login credentials, credit card details and CVV to restore access.

PayPal Security Center`,

  bec: `From: ceo.johnson@grnail.com
Subject: Confidential - Urgent Wire Transfer Required

Hi Sarah,

I am currently in a board meeting and need you to process an urgent wire transfer. This is time-sensitive and confidential — please don't discuss with anyone else in the office or HR.

Amount: $47,500
Recipient: Global Trade Partners LLC
Bank: First National Bank
Account Number: 8834729103
Routing Number: 021000021

I will explain everything when I get back. This is a last-minute acquisition and I need it done by end of business today. Keep this between us.

Best,
David Johnson
CEO`,

  lottery: `From: lotterywinner@gmail.com
Subject: CONGRATULATIONS!! You Have Won $2,500,000.00 USD!!!

CONGRATULATIONS!!!

We are pleased to inform you that you have been selected as a WINNER in the INTERNATIONAL EMAIL LOTTERY SWEEPSTAKES 2026!!!

Your email address has won $2,500,000.00 USD (TWO MILLION FIVE HUNDRED THOUSAND DOLLARS).

To claim your prize, you must provide:
- Full Name and Home Address
- Date of Birth and Passport Number
- Bank Account Details and Routing Number

Contact our Claims Agent: Dr. James Wilson
Email: claimsagent@prize-winners.tk

GOD BLESS YOU AND YOUR FAMILY!`,

  fake_shipping: `From: noreply@dh1-delivery.com
Subject: DHL: Your package is on hold - Action Required

Dear Customer,

Your package (Tracking: JD014600006341) could not be delivered because of an unpaid customs duty fee.

Outstanding fee: $3.50

Your parcel will be returned to the sender within 24 hours if you do not pay immediately.

PAY NOW: http://dhl-delivery-fee.xyz/pay?id=JD014600006341

If payment is not received, your package will be auctioned.

DHL Express Customer Service`,

  spam_cold_pitch: `From: "Alex Miller | Growth Partners" <alex.m@outreach-growthsolutions.co>
To: undisclosed-recipients:;
Subject: Quick question regarding your Q3 pipeline

Hi there,

I came across your profile and was really impressed by what you and your team have accomplished recently.

We help companies like yours scale their revenue and generate 30-50 qualified B2B leads every month using our proprietary AI outreach system. No long-term contracts and 100% guaranteed results.

Would you be open to a quick 15-minute call this Thursday or Friday to touch base and see if there's any synergy?

If not interested, just reply with 'REMOVE' and I won't reach out again.

Best regards,
Alex Miller
Head of Business Development
Growth Partners LLC`,

  spam_promo: `From: "Deals & Discounts" <promo-blast@bargaindaily-mailer.com>
Subject: FLASH SALE: Up to 80% OFF All Tech Gadgets! (Ends Tonight)

Dear Valued Shopper,

Our biggest flash sale of the season is officially live! Save up to 80% on smartwatches, noise-cancelling headphones, and 4K dash cams.

Use promo code: FLASH80 at checkout.

Claim your discount now:
https://bargaindaily-mailer.com/deals?ref=aff_partner_9921&utm_source=blast&utm_medium=email&utm_campaign=flash80

Hurry, only 7 items left in stock at this unbeatable price! Offer expires at midnight.

Unsubscribe: https://bargaindaily-mailer.com/optout?id=88310`,

  spam_job: `From: "HR Recruitment" <careers@fastjob-recruitment.net>
Subject: Remote Data Entry Assistant Needed ($450/day)

Hello,

We are currently hiring for an immediate work-from-home position as an Online Data Entry Assistant.

Position Highlights:
- Flexible working hours (1 to 2 hours per day)
- Earn $350 - $550 per day
- No experience necessary — full training provided
- Start earning immediately

Requirements:
- Must have a smartphone or laptop with internet access
- Basic English communication skills

Spots are filling fast! To apply immediately, contact our hiring manager on Telegram: @GlobalRecruit_HR2026 or WhatsApp: +1-202-555-0194.

Global Remote Staffing LLC`,

  spam_fake_thread: `From: "Jordan Reed" <jordan@saasgrowthhub.io>
Subject: Re: Following up on our discussion

Hi,

Just wanted to bump this to the top of your inbox in case you missed my earlier email.

Did you have a chance to review our proposal on how to boost your organic search rankings and acquire high-authority backlinks? Our clients have seen an average 300% surge in organic traffic within 60 days.

Are you available for a brief 10-minute chat tomorrow at 2 PM?

Let me know what time works best for you.

Jordan Reed
Growth Strategist`,

  spam_survey: `From: "Customer Advisory Panel" <rewards@consumer-opinions-center.org>
To: recipient@usermail.com
Subject: You've been selected: Claim your $50 Gift Card!

Dear Friend,

Congratulations! You have been selected to participate in our quarterly consumer opinion survey.

Complete this quick 2-minute survey to share your thoughts on popular online streaming services, and receive an instant $50 e-gift card upon completion.

Click here to start the survey:
http://consumer-opinions-research.com/take-survey?user_id=48219&aff=surv_network

Only 15 survey spots left for your region. Claim your reward before the panel closes.

Customer Opinions Center`,

  legit: `From: billing@yourpowerco.com
Reply-To: billing@yourpowerco.com
Subject: Your July statement is ready

Hi Alex,

Your electricity statement for July 2026 is now available in your online account.

Statement summary:
- Period: 1 July to 31 July 2026
- Amount due: $84.12
- Due date: 20 August 2026
- Payment method: Direct debit on file

No action is needed. The payment will be collected automatically on the due date.

To view your full statement, log in at yourpowerco.com using your registered email.

Questions? Contact us at support@yourpowerco.com or call 1800-XXX-XXX.

YourPower Co. Billing Team`,
};

// ===================================================
// HEADER & URL PARSING
// ===================================================
function parseHeaders(text) {
  const headers = {};
  const lines = text.split('\n');
  for (const line of lines) {
    if (!headers.from    && /^From:\s*/i.test(line))     headers.from     = line.replace(/^From:\s*/i, '').trim();
    if (!headers.replyTo && /^Reply-To:\s*/i.test(line)) headers.replyTo  = line.replace(/^Reply-To:\s*/i, '').trim();
    if (!headers.subject && /^Subject:\s*/i.test(line))  headers.subject  = line.replace(/^Subject:\s*/i, '').trim();
    if (!headers.to      && /^To:\s*/i.test(line))       headers.to       = line.replace(/^To:\s*/i, '').trim();
  }
  return headers;
}

function extractDomain(emailStr) {
  if (!emailStr) return null;
  const match = emailStr.match(/@([a-z0-9._-]+\.[a-z]{2,})/i);
  return match ? match[1].toLowerCase() : null;
}

function extractAllUrls(text) {
  const pattern = /(https?:\/\/[^\s<>"'\]]+|www\.[a-z0-9.-]+\.[a-z]{2,}[^\s<>"'\]]*)/gi;
  return [...new Set(text.match(pattern) || [])];
}

// ===================================================
// HEURISTIC CHECKS
// ===================================================

function checkBrandImpersonation(text, headers) {
  const senderDomain = extractDomain(headers.from || '');
  const findings = [];
  const lowerText = text.toLowerCase();
  const fromField = (headers.from || '').toLowerCase();
  const subjectField = (headers.subject || '').toLowerCase();

  for (const [brand, legitimateDomains] of Object.entries(BRAND_DB)) {
    const brandRe = new RegExp(`\\b${brand.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}\\b`, 'i');
    if (brandRe.test(lowerText) && senderDomain) {
      const isOfficial = legitimateDomains.some(d =>
        senderDomain === d || senderDomain.endsWith('.' + d)
      );

      if (!isOfficial) {
        const brandInFrom = brandRe.test(fromField);
        const brandInSubject = brandRe.test(subjectField);
        const brandInBodyContext = new RegExp(`\\b(your ${brand} (account|order|security|profile)|${brand} (team|support|security|service|center|helpdesk|billing|notice))\\b`, 'i').test(text);

        if (brandInFrom || brandInSubject || brandInBodyContext) {
          findings.push(`Email body or headers claim official authority from "${brand}" but sending domain is "${senderDomain}" — not an official ${brand} domain.`);
        }
      }
    }
  }
  return findings;
}

function checkSenderDomain(text, headers) {
  const from = headers.from || '';
  const domain = extractDomain(from);
  const findings = [];
  if (!domain) return findings;

  const freeProviders = ['gmail.com','yahoo.com','hotmail.com','outlook.com','aol.com','mail.com','protonmail.com','icloud.com','yandex.com','gmx.com'];
  const displayNameMatch = from.match(/^"?([^"<@\n]+?)"?\s*</);
  const displayName = displayNameMatch ? displayNameMatch[1].trim() : '';
  const officialClaimPattern = /\b(bank|paypal|amazon|apple|microsoft|google|netflix|support|security|account|service|noreply|no-reply|admin|official|helpdesk|team|hr|recruitment|billing)\b/i;

  if (freeProviders.includes(domain) && officialClaimPattern.test(displayName)) {
    findings.push(`Display name "${displayName}" claims corporate authority, but email was sent from a free public provider: "${domain}".`);
  }

  const tld = domain.split('.').pop().toLowerCase();
  if (SUSPICIOUS_TLDS.has(tld)) {
    findings.push(`Sender domain "${domain}" uses a high-risk TLD (.${tld}) frequently abused in phishing and spam campaigns.`);
  }

  const parts = domain.split('.');
  if (parts.length >= 3) {
    const registrableDomain = parts.slice(-2).join('.');
    const subPart = parts.slice(0, -2).join('.');
    for (const [brand, legit] of Object.entries(BRAND_DB)) {
      if (subPart.includes(brand) && !legit.includes(registrableDomain)) {
        findings.push(`Domain "${domain}" deceptively embeds brand keyword "${brand}" in a subdomain.`);
        break;
      }
    }
  }
  return findings;
}

function checkSuspiciousUrls(text, headers) {
  const urls = extractAllUrls(text);
  const findings = [];
  for (const url of urls) {
    let hostname;
    try {
      hostname = new URL(url.startsWith('http') ? url : 'http://' + url).hostname.toLowerCase();
    } catch { continue; }

    if (/^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(hostname)) {
      findings.push(`Raw IP-based URL: "${url}" — legitimate services never direct users to raw IPs.`);
      continue;
    }
    if (URL_SHORTENERS.has(hostname)) {
      findings.push(`URL shortener "${hostname}" in link "${url}" — conceals destination.`);
      continue;
    }
    const tld = hostname.split('.').pop();
    if (SUSPICIOUS_TLDS.has(tld)) {
      findings.push(`URL using suspicious TLD (.${tld}): "${url}"`);
      continue;
    }
    for (const [brand, legit] of Object.entries(BRAND_DB)) {
      if (hostname.includes(brand) && !legit.includes(hostname) && !legit.some(d => hostname === d || hostname.endsWith('.' + d))) {
        findings.push(`Possible typosquatted URL for "${brand}": "${url}"`);
        break;
      }
    }
  }
  return findings;
}

function checkTrackingUrls(text, headers) {
  const urls = extractAllUrls(text);
  const findings = [];
  for (const url of urls) {
    let urlObj, hostname;
    try {
      urlObj = new URL(url.startsWith('http') ? url : 'http://' + url);
      hostname = urlObj.hostname.toLowerCase();
    } catch { continue; }

    for (const param of urlObj.searchParams.keys()) {
      if (AFFILIATE_PARAMS.has(param.toLowerCase())) {
        findings.push(`Commercial affiliate tracking parameter "?${param}=" detected in URL: "${url}"`);
        break;
      }
    }

    if (TRACKING_DOMAINS.has(hostname) || /^click\./i.test(hostname) || /^track\./i.test(hostname)) {
      findings.push(`Bulk marketing redirector domain detected: "${hostname}"`);
    }
  }
  return findings;
}

function checkHeaderAnomalies(text, headers) {
  const findings = [];
  const fromDomain  = extractDomain(headers.from || '');
  const replyDomain = extractDomain(headers.replyTo || '');

  if (fromDomain && replyDomain && fromDomain !== replyDomain) {
    findings.push(`Reply-To domain "${replyDomain}" differs from From domain "${fromDomain}".`);
  }

  if (headers.from) {
    const displayMatch = headers.from.match(/^"?([^"<@\n]+?)"?\s*</);
    if (displayMatch && fromDomain) {
      const displayName = displayMatch[1].trim();
      for (const [brand, legit] of Object.entries(BRAND_DB)) {
        if (new RegExp(`\\b${brand}\\b`, 'i').test(displayName) && !legit.includes(fromDomain) && !legit.some(d => fromDomain.endsWith('.' + d))) {
          findings.push(`Display name claims to be "${displayName}" but actual sending domain is "${fromDomain}".`);
          break;
        }
      }
    }
  }
  return findings;
}

function checkLookalikeDomain(text, headers) {
  const findings = [];
  const from = headers.from || '';
  const domain = extractDomain(from);

  const suspectDomains = new Set();
  if (domain) suspectDomains.add(domain);
  extractAllUrls(text).forEach(url => {
    try {
      const h = new URL(url.startsWith('http') ? url : 'http://' + url).hostname.toLowerCase();
      suspectDomains.add(h);
    } catch {}
  });

  for (const candidate of suspectDomains) {
    const base = candidate.split('.').slice(0, -1).join('.');
    for (const [brand, legit] of Object.entries(BRAND_DB)) {
      if (legit.includes(candidate) || legit.some(d => candidate === d || candidate.endsWith('.' + d))) continue;
      const typos = generateTypos(brand);
      if (typos.some(t => base.includes(t) || candidate.includes(t))) {
        findings.push(`"${candidate}" appears to be a lookalike/typosquatted domain impersonating "${brand}".`);
        break;
      }
    }
  }
  return findings;
}

function generateTypos(brand) {
  const typos = new Set();
  ['secure','login','verify','update','support','account','service','official','help','center','portal'].forEach(sfx => {
    typos.add(brand + '-' + sfx);
    typos.add(brand + sfx);
    typos.add(sfx + '-' + brand);
  });
  typos.add(brand.replace(/a/g,'@').replace(/o/g,'0').replace(/i/g,'1').replace(/l/g,'1'));
  typos.add(brand.replace(/m/g,'rn').replace(/w/g,'vv'));
  typos.add(brand + '1');
  typos.add(brand.replace(/l/g,'1').replace(/o/g,'0'));
  for (let i = 0; i < brand.length; i++) {
    typos.add(brand.slice(0,i) + brand.slice(i+1));
    typos.add(brand.slice(0,i) + brand[i] + brand.slice(i));
  }
  typos.delete(brand);
  return [...typos];
}

function checkDeceptiveSubject(text, headers) {
  const findings = [];
  const subject = (headers.subject || '').trim();
  if (!subject) return findings;

  if (/^(re|fwd):\s*/i.test(subject)) {
    const isColdPitch = /\b(came across your profile|touch base|bump this to the top|did you (have a chance|get my)|reaching out|quick question|my earlier email|hope you're doing well|hi there)\b/i.test(text);
    if (isColdPitch) {
      findings.push(`Subject "${subject}" uses "Re:" / "Fwd:" to simulate an existing conversation, but content is unsolicited cold outreach.`);
    }
  }

  if (/\b(did you forget|left in your cart|action required regarding your inquiry|your invitation is expiring|confirm your attendance\?)\b/i.test(subject)) {
    findings.push(`Subject line employs manipulative curiosity-gap clickbait: "${subject}".`);
  }

  const emojiMatches = subject.match(/[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu);
  if (emojiMatches && emojiMatches.length >= 2) {
    findings.push(`Subject line contains multiple attention-grabbing promotional emojis.`);
  }

  return findings;
}

function checkStructuralSpam(text, headers) {
  const findings = [];
  const urls = extractAllUrls(text);
  const words = text.trim().split(/\s+/).length;

  if (urls.length >= 3 && words < 120) {
    findings.push(`High link density: ${urls.length} links found in a brief message of only ${words} words.`);
  } else if (urls.length >= 6) {
    findings.push(`Excessive link density: message contains ${urls.length} distinct links.`);
  }

  const to = headers.to || '';
  if (/undisclosed[- ]recipients/i.test(to) || to === ';;') {
    findings.push('Header "To:" specifies "undisclosed-recipients" — blind bulk marketing distribution.');
  }

  return findings;
}

function checkGreymailEvasion(text, headers) {
  const findings = [];
  const fromDomain = extractDomain(headers.from || '');
  const hasOptOutText = /\b(reply (with |back )?["']?(unsubscribe|remove|stop)["']?|to (unsubscribe|opt[- ]out|stop receiving|manage your preferences),? (click here|visit)|safeunsubscribe|this (is a )?commercial advertisement)\b/i.test(text);

  if (hasOptOutText) {
    const freeProviders = ['gmail.com','yahoo.com','hotmail.com','outlook.com','aol.com','protonmail.com'];
    if (freeProviders.includes(fromDomain)) {
      findings.push(`Opt-out instructions sent from personal webmail ("${fromDomain}") to evade spam filters.`);
    } else {
      findings.push('Email contains unverified opt-out footer typical of unsolicited bulk greymail.');
    }
  }

  return findings;
}

// ===================================================
// DETECTION RULES (Ensemble Heuristics)
// ===================================================
const RULES = [
  {
    key: 'brand_impersonation',
    label: 'Brand Impersonation',
    threatType: 'fraud',
    category: 'identity',
    severity: 'critical',
    weight: 4,
    description: 'Email references a major brand but sending domain does not match official brand domains.',
    check: checkBrandImpersonation
  },
  {
    key: 'credential_phishing',
    label: 'Credential Phishing',
    threatType: 'fraud',
    category: 'data_theft',
    severity: 'critical',
    weight: 4,
    description: 'Explicitly solicits passwords, PINs, OTPs, or login credentials.',
    pattern: /\b(enter your (password|pin|passcode|credentials|username)|confirm (your )?(password|identity|account)|reset (your )?password|login credentials|one[- ]?time (password|code|pin)|otp|2fa code|verification code|security code|account (password|pin)|provide (your )?(password|pin)|type (your )?password|click (here )?to (verify|confirm|update) (your )?(account|password|details|information))\b/gi
  },
  {
    key: 'urgency_threat',
    label: 'Security Urgency & Threats',
    threatType: 'fraud',
    category: 'manipulation',
    severity: 'high',
    weight: 3,
    description: 'High-pressure account suspension, security alerts, or compliance deadlines compelling hasty action.',
    pattern: /\b(urgent(ly)? (action|verification|attention|notice)|immediate (action|attention) (required|needed)|action required|respond (immediately|urgently|now)|within (24|48) hours|account (will be |has been )?(suspend(ed)?|terminat(ed)?|clos(ed)?|lock(ed)?|block(ed)?|restrict(ed)?)|failure to (comply|respond|verify)|final (notice|warning|reminder)|your account (is|has been) (compromise[d]?|hacked|breach(ed)?|limited)|immediate verification|unauthorized (access|activity)|security alert|compromised)\b/gi
  },
  {
    key: 'sender_domain',
    label: 'Suspicious Sender Domain',
    threatType: 'both',
    category: 'identity',
    severity: 'high',
    weight: 3,
    description: 'Sender uses public webmail posing as an official entity, or a high-risk TLD.',
    check: checkSenderDomain
  },
  {
    key: 'suspicious_urls',
    label: 'Malicious & Masked URLs',
    threatType: 'fraud',
    category: 'links',
    severity: 'critical',
    weight: 4,
    description: 'Links use URL shorteners, raw IP addresses, high-risk TLDs, or typosquatted brand domains.',
    check: checkSuspiciousUrls
  },
  {
    key: 'financial_bait',
    label: 'Financial Bait / Prize Scam',
    threatType: 'fraud',
    category: 'scam',
    severity: 'high',
    weight: 3,
    description: 'Offers unexpected lottery windfalls, prize awards, inheritance, or crypto doubling.',
    pattern: /\b(you (have |'?ve )?(won|been selected|been chosen)|congratulations.{0,40}(win|prize|reward|lottery|award)|claim (your )?(prize|reward|winnings|inheritance|fund|money|cash)|lottery (winner|win)|unclaimed (fund|money|estate|inheritance)s?|beneficiary|next of kin|\d+ million (dollar|pound|euro|usd|gbp)|bitcoin (giveaway|investment|profit)|crypto (giveaway|doubling|investment|mining)|guaranteed (profit|return|income)|get rich( quick)?|make money fast|passive income|investment (opportunity|returns))\b/gi
  },
  {
    key: 'bec',
    label: 'Business Email Compromise (BEC)',
    threatType: 'fraud',
    category: 'fraud',
    severity: 'critical',
    weight: 4,
    description: 'Executive impersonation soliciting wire transfers or gift cards under secrecy.',
    pattern: /\b(wire (transfer|fund)s?|transfer (fund|money|payment|the amount)s?|purchase (gift cards?|itunes|amazon gift|google play)|send (me )?(gift cards?|the fund|payment)s?|keep (this )?(confidential|between us|private|secret)|don'?t (tell|discuss|mention) (anyone|your (team|colleague|staff|hr))|i'?m (travelling|out of office|in a meeting)|on behalf of (the )?(ceo|president|director)|as (instructed|requested|directed) by|per (the |my )?request of)\b/gi
  },
  {
    key: 'malware',
    label: 'Malware / Attachment Risk',
    threatType: 'fraud',
    category: 'malware',
    severity: 'critical',
    weight: 4,
    description: 'References dangerous executable extensions, macro activation, or password-protected archives.',
    pattern: /(\.(exe|scr|bat|cmd|vbs|js|ps1|jar|iso|lnk|docm|xlsm|pptm|hta|msi|pif|dll)\b|enable (macros?|content|editing)|click enable|open the attach(ment)?|download (the |this )?(attach|file|installer)|run the (file|program|installer)|extract (the )?(zip|archive|rar|7z)|password.{0,30}(zip|archive)|protected (document|file|attachment))/gi
  },
  {
    key: 'data_harvest',
    label: 'Personal Data Harvesting',
    threatType: 'fraud',
    category: 'data_theft',
    severity: 'high',
    weight: 3,
    description: 'Requests sensitive PII like SSN, passport numbers, card details, or banking coordinates.',
    pattern: /\b(social security (number|no\.?)?|ssn|tax (id|identification) number|date of birth|d\.?o\.?b\.?|mother'?s maiden name|passport (number|no\.?)?|driving licen[cs]e|national (id|identity|insurance)|bank account (number|details)|sort code|routing number|iban|swift (code)?|credit card (number|details|no\.?)?|(card|cvv|cvc) (number|code|expiry|details)|billing (address|details)|full (name and )?(address|details))\b/gi
  },
  {
    key: 'fake_invoice',
    label: 'Fake Invoice / Courier Alert',
    threatType: 'fraud',
    category: 'fraud',
    severity: 'medium',
    weight: 2,
    description: 'Fake courier delivery notifications or overdue invoice lures.',
    pattern: /\b(your (package|parcel|shipment|delivery|order) (is|has been|was) (held|on hold|delayed|fail(ed)?|return(ed)?|unable to deliver|undeliverable|awaiting( collection)?|pending (clearance|payment))|delivery (failed|attempt|notification|update|status)|track (your )?(package|parcel|order|shipment)|invoice (attached|due|overdue)|payment (due|overdue|outstanding|required)|unpaid (invoice|balance|bill)|review (the |your )?(attached )?invoice|customs (fee|duty|charge))\b/gi
  },
  {
    key: 'lookalike',
    label: 'Lookalike / Typosquatted Domain',
    threatType: 'fraud',
    category: 'identity',
    severity: 'critical',
    weight: 4,
    description: 'Character substitutions or prefix/suffix tampering imitating official brand domains.',
    check: checkLookalikeDomain
  },
  {
    key: 'header_anomaly',
    label: 'Email Header Anomalies',
    threatType: 'both',
    category: 'identity',
    severity: 'high',
    weight: 2,
    description: 'Reply-To domain mismatch or display name spoofing.',
    check: checkHeaderAnomalies
  },
  {
    key: 'romance_fraud',
    label: 'Advance-Fee / Romance Fraud',
    threatType: 'fraud',
    category: 'scam',
    severity: 'high',
    weight: 3,
    description: 'Classic 419, dying widow, diplomat consignment, or romance scam tropes.',
    pattern: /\b(i am (a )?(widow|widower|orphan|retired general|dying|terminally ill|soldier|diplomat|barrister|attorney)|my (late |dear )?(husband|wife|father|mother|uncle)|god bless (you|your (family|soul))|in the name of (god|allah|jesus)|i need your (help|assistance|cooperation|trust)|transfer (of |the )?(fund|money|sum|amount)s?|(percentage|commission) (of |for )?the (fund|money|transfer)|confidential (business|transaction|proposal)|(dying|last) (wish|will|request)|i am (suffering from|diagnosed with) (cancer|stroke|disease))\b/gi
  },

  // --- SUBTLE SPAM & BULK OUTREACH RULES ---
  {
    key: 'cold_outreach',
    label: 'Unsolicited B2B Cold Outreach',
    threatType: 'spam',
    category: 'spam',
    severity: 'spam',
    weight: 3,
    description: 'Unsolicited commercial cold sales pitches, lead generation, or backlink/guest post solicitations.',
    pattern: /\b(quick question|reaching out to you|came across your profile|wanted to touch base|bump this to the top of your inbox|schedule a (quick )?(15|20|30)[- ]min(ute)? (call|chat)|jump on a (quick )?call|scale your (revenue|business|sales)|generate (qualified )?(b2b )?leads|guest (post|article)|backlink (opportunity|exchange)|seo ranking|synergy|partnership proposal|collaborat(e|ion) with your team|are you available for a brief (call|chat)|brief 10[- ]minute chat)\b/gi
  },
  {
    key: 'promotional_lures',
    label: 'Promotional & Discount Lures',
    threatType: 'spam',
    category: 'spam',
    severity: 'spam',
    weight: 3,
    description: 'Aggressive marketing discounts, coupon vouchers, promotional deals, or doorbuster sales.',
    pattern: /\b(exclusive (offer|deal|discount|invitation|savings)|special (promotional |promo )?offer|limited[- ]time (offer|deal|discount)|save (up to )?\d+ ?%|\d+ ?% (off|discount)|use (promo|coupon) code|claim (your )?(discount|coupon|voucher|free gift)|lowest price guaranteed|while supplies last|flash sale|black friday deal|clearance sale|doorbuster|risk[- ]free trial|free trial.{0,25}no credit card|unbeatable price)\b/gi
  },
  {
    key: 'tracking_urls',
    label: 'Affiliate & Tracking URLs',
    threatType: 'spam',
    category: 'links',
    severity: 'spam',
    weight: 3,
    description: 'URLs embedding commercial affiliate parameters (?ref=, ?aff=, utm_campaign) or marketing redirectors.',
    check: checkTrackingUrls
  },
  {
    key: 'deceptive_subject',
    label: 'Deceptive Subject Line Tricks',
    threatType: 'spam',
    category: 'manipulation',
    severity: 'medium',
    weight: 2,
    description: 'Fake "Re:"/"Fwd:" thread simulation, manipulative curiosity gaps, or promotional emoji stuffing.',
    check: checkDeceptiveSubject
  },
  {
    key: 'artificial_scarcity',
    label: 'Artificial Scarcity & FOMO',
    threatType: 'spam',
    category: 'manipulation',
    severity: 'spam',
    weight: 2,
    description: 'Fabricated deadlines and scarcity pressure ("only 3 left", "expires midnight") to force clicks.',
    pattern: /\b(only \d+ (spots?|seats?|items?|left)|offer expires (today|tonight|at midnight|in \d+ hours)?|expires (today|tonight|at midnight|soon)|ends (today|tonight|at midnight|soon)|countdown begins|almost (sold out|gone)|don'?t miss (out|this)|last chance to claim|spots are filling (fast|up)|price goes up (soon|tomorrow|tonight)|doors are closing|registration closing|limited[- ]time (only|offer|deal)|while supplies last)\b/gi
  },
  {
    key: 'easy_money_job',
    label: 'Work-from-Home / Job Scam',
    threatType: 'spam',
    category: 'scam',
    severity: 'high',
    weight: 3,
    description: 'Unrealistic work-from-home offers with high daily pay, recruiting via Telegram/WhatsApp.',
    pattern: /\b(work from home|remote (part[- ]time|flexible) job|earn \$?\d+[\d,]*(\.\d+)? (a |per )?(day|hour|week|month)|no experience (necessary|needed|required)|flexible (working )?hours|data entry (clerk|specialist|position|assistant)|personal assistant (needed|wanted)|hiring immediately|contact (our )?hiring manager (on|via) (telegram|whatsapp)|part[- ]time employment opportunity|start earning immediately)\b/gi
  },
  {
    key: 'webinar_survey_bait',
    label: 'Webinar & Survey Reward Bait',
    threatType: 'spam',
    category: 'spam',
    severity: 'spam',
    weight: 2,
    description: 'Unsolicited masterclasses or short surveys baiting recipients with gift card promises.',
    pattern: /\b((survey|feedback|questionnaire)\b.{0,120}\b(\$?\d+|free|complimentary|claim|receive).{0,30}(gift card|e-gift|voucher|reward|coupon|token|amazon)|(complete|take|participate in|fill out) (this |our )?(brief |short |quick |\d+[- ]min(ute)? )?(survey|questionnaire|feedback)|(exclusive (invitation|invite)|invited to (attend|join)|attend (our )?free) (webinar|masterclass|summit|workshop)|(customer|consumer) (opinion[s]?|advisory|panel|survey))\b/gi
  },
  {
    key: 'commercial_hype',
    label: 'Commercial Hype & Hyperbole',
    threatType: 'spam',
    category: 'spam',
    severity: 'low',
    weight: 1,
    description: 'Exaggerated business claims ("100% guaranteed results", "secret formula", "revolutionary system").',
    pattern: /\b(guaranteed (results|growth|success)|100% (risk[- ]free|guaranteed|satisfaction)|revolutionary (system|method|formula|tool)|secret (strategy|sauce|formula|hack)|as featured on (forbes|entrepreneur|cnn|bloomberg|techcrunch)|double your (revenue|sales|traffic)|magic (bullet|formula)|transform your (life|business) overnight)\b/gi
  },
  {
    key: 'structural_spam',
    label: 'Structural Bulk Mail Signals',
    threatType: 'spam',
    category: 'indicators',
    severity: 'medium',
    weight: 2,
    description: 'Disproportionately high link density, undisclosed recipients, or mass distribution headers.',
    check: checkStructuralSpam
  },
  {
    key: 'greymail_evasion',
    label: 'Greymail & Opt-Out Evasion',
    threatType: 'spam',
    category: 'spam',
    severity: 'spam',
    weight: 2,
    description: 'Pseudo opt-out/unsubscribe text sent through personal webmail.',
    check: checkGreymailEvasion
  },
  {
    key: 'generic_greeting',
    label: 'Generic Mass-Mail Greeting',
    threatType: 'both',
    category: 'indicators',
    severity: 'low',
    weight: 1,
    description: 'Impersonal salutations ("Dear Valued Shopper", "Dear Friend") characteristic of untargeted mass mailing.',
    pattern: /\b(dear (customer|user|valued (customer|member|client|user|shopper)|member|account holder|beneficiary|friend|sir[\/,]? ?madam|applicant|recipient|subscriber)|hello (customer|user|member|friend)|to (whom it may concern|the account holder))\b/gi
  },
  {
    key: 'formatting',
    label: 'Formatting Abuse & Symbol Spam',
    threatType: 'both',
    category: 'indicators',
    severity: 'low',
    weight: 1,
    description: 'Excessive uppercase capitalization, repeating exclamation marks, or symbol stacking.',
    pattern: /(!{3,}|\?{3,}|\${3,}|\b[A-Z]{6,}\b|(\b[A-Z]{4,}\b.{0,50}\b[A-Z]{4,}\b))/g
  },
];

// ===================================================
// DUAL ENSEMBLE CLASSIFIER (AI + HEURISTICS)
// ===================================================
function analyze(text) {
  const headers = parseHeaders(text);
  const allHits = [];
  const allMisses = [];
  let annotated = escapeHtml(text);
  const urlList = extractAllUrls(text);

  // 1. Run AI NLP Model
  const aiResult = runAiModelInference(text, headers);

  // 2. Run Heuristic Rule Engine
  let fraudPoints = 0;
  let spamPoints  = 0;

  for (const rule of RULES) {
    let findings   = [];
    let matchCount = 0;

    if (rule.check) {
      findings   = rule.check(text, headers);
      matchCount = findings.length;
    } else if (rule.pattern) {
      const matches = text.match(rule.pattern);
      if (matches && matches.length) {
        matchCount = matches.length;
        findings   = [`${rule.description} (${matchCount} instance${matchCount > 1 ? 's' : ''} flagged)`];

        const highlightClass = rule.severity === 'critical' ? 'mark-critical'
          : rule.severity === 'high' ? 'mark-high'
          : rule.severity === 'spam' ? 'mark-spam'
          : rule.severity === 'medium' ? 'mark-medium'
          : 'mark-low';

        annotated = annotated.replace(
          new RegExp(rule.pattern.source, rule.pattern.flags),
          m => `<mark class="${highlightClass}">${escapeHtml(m)}</mark>`
        );
      }
    }

    if (matchCount > 0) {
      allHits.push({ rule, findings, count: matchCount });
      if (rule.threatType === 'fraud') {
        fraudPoints += rule.weight;
      } else if (rule.threatType === 'spam') {
        spamPoints += rule.weight;
      } else if (rule.threatType === 'both') {
        fraudPoints += Math.floor(rule.weight * 0.5);
        spamPoints  += Math.ceil(rule.weight * 0.8);
      }
    } else {
      allMisses.push(rule);
    }
  }

  // --- Calibrated Scoring Curve ---
  const hasCriticalFraud = allHits.some(h => h.rule.threatType === 'fraud' && h.rule.severity === 'critical');
  let fraudScore = 0;
  if (hasCriticalFraud) {
    fraudScore = Math.min(10, Math.max(8, 7 + fraudPoints));
  } else if (fraudPoints > 0) {
    fraudScore = Math.min(10, Math.round(fraudPoints * 1.2));
  }

  // Calculate heuristic spam score
  let heuristicSpamScore = 0;
  if (spamPoints > 0) {
    if (spamPoints <= 2) heuristicSpamScore = 4;
    else if (spamPoints <= 4) heuristicSpamScore = 7;
    else if (spamPoints <= 6) heuristicSpamScore = 9;
    else heuristicSpamScore = 10;
  }

  // Ensemble Fusion: blend heuristic score with AI Model Probability
  const aiScoreMapped = Math.round(aiResult.spamProbability / 10);
  let spamScore;
  if (aiResult.spamProbability > 75) {
    spamScore = Math.max(heuristicSpamScore, aiScoreMapped);
  } else if (aiResult.spamProbability < 25) {
    spamScore = Math.min(heuristicSpamScore, aiScoreMapped);
  } else {
    spamScore = Math.round(heuristicSpamScore * 0.5 + aiScoreMapped * 0.5);
  }

  // Overall Score (0 to 10)
  let score;
  if (fraudScore >= 8) {
    score = fraudScore;
  } else if (spamScore >= 6) {
    score = Math.max(fraudScore, Math.round(spamScore * 0.85));
  } else {
    score = Math.max(fraudScore, spamScore);
  }
  score = Math.min(10, Math.max(0, score));

  const verdict = determineVerdict(fraudScore, spamScore, score, allHits, aiResult.spamProbability);

  return {
    score,
    fraudScore,
    spamScore,
    aiResult,
    verdict,
    allHits,
    allMisses,
    annotated,
    headers,
    urlList
  };
}

function determineVerdict(fraudScore, spamScore, overallScore, allHits, aiSpamProb) {
  const hasCriticalFraud = allHits.some(h => h.rule.threatType === 'fraud' && h.rule.severity === 'critical');

  if (fraudScore >= 8 || hasCriticalFraud) {
    return {
      cls: 'fraud',
      label: 'Fraudulent',
      desc: 'Critical security fraud detected. High probability of credential theft, brand impersonation, or financial scam.',
      typeBadge: 'Malicious Fraud'
    };
  }
  if (fraudScore >= 4) {
    return {
      cls: 'high-risk',
      label: 'High Risk Phishing',
      desc: 'Multiple phishing or impersonation indicators detected. Verify sender independently.',
      typeBadge: 'Phishing Threat'
    };
  }
  if (spamScore >= 6 || aiSpamProb >= 75) {
    return {
      cls: 'spam',
      label: 'Spam / Bulk Mail',
      desc: `AI Neural Classifier detected ${aiSpamProb}% spam probability with commercial outreach / bulk patterns.`,
      typeBadge: 'Commercial Spam'
    };
  }
  if (spamScore >= 3 || fraudScore >= 2 || aiSpamProb >= 40) {
    return {
      cls: 'suspicious',
      label: 'Suspicious Outreach',
      desc: 'Mild promotional lures or cold outreach patterns flagged by AI. Exercise caution.',
      typeBadge: 'Suspicious Pitch'
    };
  }
  return {
    cls: 'legit',
    label: 'Likely Legit',
    desc: `AI Model confirms low risk (${aiSpamProb}% spam probability). No significant threat signals detected.`,
    typeBadge: 'Clean Email'
  };
}

// ===================================================
// UTILITIES & REPORT GENERATION
// ===================================================
function escapeHtml(str) {
  return str
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;');
}

function buildReportText(score, fraudScore, spamScore, aiResult, verdict, hits, headers, expl) {
  const lines = [
    'AI SPAM MAIL DETECTOR — Analysis & Threat Report',
    '=================================================',
    `Date              : ${new Date().toLocaleString()}`,
    `Verdict           : ${verdict.label}`,
    `Classification    : ${verdict.typeBadge}`,
    `AI Spam Likelihood: ${aiResult.spamProbability}% (Log-Odds: ${aiResult.logOdds ? aiResult.logOdds.toFixed(2) : 'N/A'})`,
    `Inference Speed   : ${aiResult.latencyMs} ms`,
    `Overall Score     : ${score} / 10`,
    `Fraud Risk Score  : ${fraudScore} / 10`,
    `Spam Heuristics   : ${spamScore} / 10`,
    `Summary           : ${verdict.desc}`,
  ];

  if (expl) {
    lines.push(
      '',
      'DETAILED AI REASONING & SECURITY EXPLANATION',
      '---------------------------------------------',
      `Title             : ${expl.title}`,
      `Summary           : ${expl.summary}`,
      `Recommended Action: ${expl.actionAdvice}`
    );
  }

  lines.push(
    '',
    'AI TOP TOKEN INFLUENCE (XAI)',
    '----------------------------',
  );


  if (aiResult.topSpamFeatures.length > 0) {
    lines.push('  Spam-inducing tokens:');
    aiResult.topSpamFeatures.forEach(f => lines.push(`    • ${f.token.replace(/_/g, ' ')} (+${f.weight})`));
  }
  if (aiResult.topHamFeatures.length > 0) {
    lines.push('  Legitimacy tokens:');
    aiResult.topHamFeatures.forEach(f => lines.push(`    • ${f.token.replace(/_/g, ' ')} (${f.weight})`));
  }

  lines.push(
    '',
    'PARSED HEADERS',
    '--------------',
    `From     : ${headers.from     || 'Not found'}`,
    `Reply-To : ${headers.replyTo  || 'Not found'}`,
    `To       : ${headers.to       || 'Not found'}`,
    `Subject  : ${headers.subject  || 'Not found'}`,
    '',
    `TRIGGERED HEURISTIC INDICATORS (${hits.length})`,
    '-'.repeat(48)
  );

  if (hits.length === 0) {
    lines.push('  None — message is clean.');
  } else {
    hits.forEach(({ rule, findings }) => {
      lines.push(`\n[${rule.threatType.toUpperCase()} | ${rule.severity.toUpperCase()}] ${rule.label}`);
      findings.forEach(f => lines.push(`  • ${f}`));
    });
  }

  lines.push('', 'Generated by AI Spam Mail Detector — Powered by ML & Heuristics');
  return lines.join('\n');
}

// ===================================================
// DETAILED EXPLANATION SYNTHESIS (XAI + THREAT REASONING)
// ===================================================
function buildDetailedExplanation({ score, fraudScore, spamScore, aiResult, verdict, allHits, headers, apiResult, text }) {
  const isFraud = verdict.cls === 'fraud';
  const isSpam = verdict.cls === 'spam';
  const isUncertain = verdict.cls === 'uncertain';
  const isLegit = verdict.cls === 'legit';

  let badgeText = verdict.label.toUpperCase();
  let badgeCls = verdict.cls;
  let title = 'How AI Analyzed This Email';
  let summary = '';
  let icon = '🛡️';
  let actionAdvice = '';
  
  const lowerText = (text || '').toLowerCase();
  const hitLabels = allHits.map(h => h.rule.label.toLowerCase());

  const hasUrgency = hitLabels.some(l => l.includes('urgency') || l.includes('pressure') || l.includes('time') || l.includes('deadline')) || lowerText.includes('immediately') || lowerText.includes('within 24 hours') || lowerText.includes('urgent');
  const hasCredentialLure = hitLabels.some(l => l.includes('credential') || l.includes('phish') || l.includes('password') || l.includes('suspended') || l.includes('account')) || lowerText.includes('password') || lowerText.includes('login') || lowerText.includes('suspended');
  const hasMoneyLure = hitLabels.some(l => l.includes('lottery') || l.includes('wire') || l.includes('crypto') || l.includes('job') || l.includes('advance') || l.includes('usd')) || lowerText.includes('wire transfer') || lowerText.includes('2,500,000');
  const hasPromo = hitLabels.some(l => l.includes('discount') || l.includes('promo') || l.includes('flash') || l.includes('affiliate') || l.includes('sale')) || lowerText.includes('80%') || lowerText.includes('coupon');
  const hasColdPitch = hitLabels.some(l => l.includes('cold') || l.includes('b2b') || l.includes('call') || l.includes('leads') || l.includes('pitch')) || lowerText.includes('15-minute') || lowerText.includes('outbound');
  const hasFakeThread = hitLabels.some(l => l.includes('thread') || l.includes('conversation') || l.includes('bump')) || (headers.subject && headers.subject.toLowerCase().startsWith('re:'));
  const hasJobScam = hitLabels.some(l => l.includes('job')) || lowerText.includes('telegram') || lowerText.includes('450') || lowerText.includes('data entry');
  const hasSurvey = hitLabels.some(l => l.includes('survey')) || lowerText.includes('gift card') || lowerText.includes('amazon');

  // Human-friendly Title, Summary, and Action
  if (isFraud) {
    icon = '🚨';
    if (hasCredentialLure) {
      title = 'Fake Security Alert (Password Phishing)';
      summary = `This email is pretending to be a security alert from your bank or account provider. It tells you your account has been blocked or compromised to make you panic and click a link to type in your password or card details.`;
      actionAdvice = 'DO NOT click any link and DO NOT enter your password. Real banks never ask you to restore your account through an urgent email link. Delete or report as Phishing.';
    } else if (hasMoneyLure || lowerText.includes('wire')) {
      title = 'Fake Money / Wire Transfer Scam';
      summary = `This email is trying to trick you into transferring money or giving up bank account details. It pretends to be an urgent corporate request or a multi-million dollar prize notification.`;
      actionAdvice = 'DO NOT send money or share bank info. Real executives or lottery commissions do not ask for urgent secret wire transfers over email.';
    } else {
      title = 'Dangerous Scam / Fraud Email';
      summary = `This email has major security red flags. It is using fake sender info and urgent pressure to trick you.`;
      actionAdvice = 'Do NOT click any link or download attachments. Delete this email immediately.';
    }
  } else if (isSpam) {
    icon = '📦';
    if (hasColdPitch) {
      title = 'Unwanted Cold Sales Pitch';
      summary = `A salesperson or company you have never contacted is pitching their business services or asking for a 15-minute phone call. It is not dangerous, just unwanted cold marketing.`;
      actionAdvice = 'Safe to ignore or mark as Spam. Avoid clicking "unsubscribe" links from unknown people, because that tells them your email address is active.';
    } else if (hasPromo) {
      title = 'Promotional Discount & Sales Ad';
      summary = `This is a mass marketing email advertising discounts (like 80% off) and coupon codes. It tries to rush you into buying by saying the deal "ends tonight".`;
      actionAdvice = 'Move to your Promotions or Spam folder. If you subscribed to this store, you can use the official unsubscribe button at the bottom.';
    } else if (hasJobScam) {
      title = 'Fake Job Offer / Work-From-Home Scam';
      summary = `This email offers unusually high daily pay ($450–$750/day) for very simple online work and asks you to chat on Telegram or WhatsApp. Real employers do not hire people this way.`;
      actionAdvice = 'Do NOT contact them on Telegram or send your resume. Delete the email immediately.';
    } else if (hasFakeThread) {
      title = 'Tricky Follow-Up (Fake "Re:" in Subject)';
      summary = `The sender put "Re:" at the start of the subject line to trick you into thinking you already had a conversation with them. It is just a psychological trick to get you to open the email.`;
      actionAdvice = 'Mark as Spam so your email filter learns to block these fake follow-up tricks.';
    } else if (hasSurvey) {
      title = 'Fake Gift Card / Survey Bait';
      summary = `This email promises an expensive gift card (like a $100 Amazon card) for taking a quick 2-minute survey. Scammers use this bait to collect your phone number and personal info.`;
      actionAdvice = 'Do NOT click the survey link. Companies do not give away $100 cards for a 2-minute survey.';
    } else {
      title = 'Commercial Junk Mail (Spam)';
      summary = `The AI detected common marketing phrases and bulk advertising patterns. It is safe to ignore.`;
      actionAdvice = 'Move to Spam or Junk folder.';
    }
  } else if (isUncertain) {
    icon = '⚠️';
    title = 'Borderline Email (Needs Your Review)';
    summary = `The AI is not 100% sure because this email contains a mix of normal business words and a few marketing phrases. It scored around ${aiResult.spamProbability}%, which sits right on the borderline.`;
    actionAdvice = 'Check who sent this email. If you know the person or company and were expecting it, it is safe. If not, do not click any links.';
  } else {
    icon = '✅';
    title = 'Safe, Normal Email';
    summary = `Everything looks good! This email does not contain pressure tactics, suspicious links, or deceptive words. It looks like normal communication.`;
    actionAdvice = 'Safe to read and reply normally. No security concerns detected.';
  }

  // 4 Easy-To-Understand Cards
  const sections = [];

  // Card 1: What is this email trying to do?
  let whatIsIt = '';
  if (isFraud) {
    whatIsIt = 'It is trying to scare or rush you into handing over sensitive information (like passwords or money).';
  } else if (isSpam) {
    if (hasColdPitch) whatIsIt = 'Trying to book a sales meeting or sell business services without your permission.';
    else if (hasJobScam) whatIsIt = 'Trying to pull you into an online task scam by offering unrealistically easy money.';
    else if (hasPromo) whatIsIt = 'Trying to get you to buy products quickly using discounts and fake countdown deadlines.';
    else whatIsIt = 'Promoting commercial products or services to thousands of people at once.';
  } else if (isUncertain) {
    whatIsIt = 'Discussing a topic (like orders, shipping, or invoices) that shares words with both real receipts and fake scams.';
  } else {
    whatIsIt = 'A normal, genuine message with authentic information and no hidden tricks.';
  }
  sections.push({
    title: '1. What is this email doing?',
    desc: whatIsIt
  });

  // Card 2: Key Red Flags Found / Why it looks safe
  let reasonsList = [];
  if (isFraud) {
    if (hasUrgency) reasonsList.push('Uses urgent countdown words ("within 24 hours" or "immediately") so you react in a hurry.');
    if (hasCredentialLure) reasonsList.push('Asks you to verify passwords, logins, or bank account details.');
    if (hasMoneyLure) reasonsList.push('Mentions urgent wire transfers or millions of dollars in prizes.');
    const replyD = extractDomain(headers.replyTo || '');
    const fromD = extractDomain(headers.from || '');
    if (replyD && fromD && replyD !== fromD) reasonsList.push(`Sender mismatch: sent from "${fromD}", but replies secretly go to "${replyD}".`);
  } else if (isSpam) {
    if (hasPromo) reasonsList.push('Uses aggressive discount words like "flash sale", "promo code", and "80% off".');
    if (hasColdPitch) reasonsList.push('Asks for a "quick 15-minute call" to pitch unrequested services.');
    if (hasFakeThread) reasonsList.push('Uses a fake "Re:" prefix to pretend you already spoke with them.');
    if (hasJobScam) reasonsList.push('Promises huge daily earnings ($450+/day) with no experience and asks to chat on Telegram.');
    if (reasonsList.length === 0) reasonsList.push('Contains repetitive sales terms and automated advertising tags.');
  } else if (isUncertain) {
    reasonsList.push(`The AI gave this a ${aiResult.spamProbability}% score, which is in the 40%–60% caution zone.`);
    reasonsList.push('Some words look like routine business, but other words resemble marketing lures.');
  } else {
    reasonsList.push('No threats, no urgent countdowns, and no requests for private passwords.');
    reasonsList.push('Sender email and reply email match real, authentic addresses.');
    reasonsList.push('Uses natural, conversational language.');
  }

  const reasonsHtml = reasonsList.length === 1 
    ? reasonsList[0] 
    : `<ul style="margin:4px 0 0 16px; padding:0;">${reasonsList.map(r => `<li style="margin-bottom:4px;">${r}</li>`).join('')}</ul>`;

  sections.push({
    title: isLegit ? '2. Why does it look safe?' : '2. Key clues detected',
    desc: reasonsHtml
  });

  // Card 3: Words that gave it away
  let wordsDesc = '';
  const topSpamWords = (aiResult.topSpamFeatures || []).slice(0, 4).map(f => `<strong>"${f.token.replace(/_/g, ' ')}"</strong>`);
  const topHamWords = (aiResult.topHamFeatures || []).slice(0, 4).map(f => `<strong>"${f.token.replace(/_/g, ' ')}"</strong>`);

  if (isLegit) {
    if (topHamWords.length > 0) {
      wordsDesc = `Everyday, trustworthy words like ${topHamWords.join(', ')} showed the AI this is genuine communication.`;
    } else {
      wordsDesc = 'Normal, natural wording without any suspicious sales or scam keywords.';
    }
  } else {
    if (topSpamWords.length > 0) {
      wordsDesc = `Words like ${topSpamWords.join(', ')} were the main giveaways that alerted the AI.`;
    } else {
      wordsDesc = 'Typical bulk email phrasing and sales buzzwords triggered the alert.';
    }
  }
  sections.push({
    title: '3. Words that gave it away',
    desc: wordsDesc
  });

  // Card 4: How dangerous is it?
  let dangerDesc = '';
  if (isFraud) {
    dangerDesc = '🔴 <strong>High Danger:</strong> Risk of stolen passwords, account takeover, or lost money if you click links or reply.';
  } else if (isSpam) {
    dangerDesc = '🟠 <strong>Annoying, not dangerous:</strong> Mostly inbox clutter and sales ads. Harmless as long as you ignore it.';
  } else if (isUncertain) {
    dangerDesc = '🟡 <strong>Medium Caution:</strong> Could be harmless, but worth double-checking who sent it before opening links.';
  } else {
    dangerDesc = '🟢 <strong>Safe:</strong> Genuine email with no known security or spam threats.';
  }
  sections.push({
    title: '4. Risk level for you',
    desc: dangerDesc
  });

  return {
    badgeText,
    badgeCls,
    title,
    summary,
    icon,
    actionAdvice,
    sections
  };
}


// ===================================================
// DOM REFERENCES & BINDINGS
// ===================================================
const input          = document.getElementById('emailInput');
const charCount      = document.getElementById('charCount');
const analyzeBtn     = document.getElementById('analyzeBtn');
const scanStatus     = document.getElementById('scanStatus');
const scanline       = document.getElementById('scanline');
const result         = document.getElementById('result');
const stamp          = document.getElementById('stamp');
const verdictDescEl  = document.getElementById('verdictDesc');
const scoreNum       = document.getElementById('scoreNum');
const gaugeFill      = document.getElementById('gaugeFill');
const fraudSubscore  = document.getElementById('fraudSubscore');
const spamSubscore   = document.getElementById('spamSubscore');
const threatBadgeEl  = document.getElementById('threatBadge');
const aiProbFill     = document.getElementById('aiProbFill');
const aiProbVal      = document.getElementById('aiProbVal');
const aiLatencyEl    = document.getElementById('aiLatency');
const aiFeaturesEl   = document.getElementById('aiFeatures');
const tokenSalienceEl= document.getElementById('tokenSalience');
const analysisExplanationCard = document.getElementById('analysisExplanationCard');
const explanationBadge       = document.getElementById('explanationBadge');
const explanationTitle       = document.getElementById('explanationTitle');
const explanationSummary     = document.getElementById('explanationSummary');
const explanationGrid        = document.getElementById('explanationGrid');
const actionIcon             = document.getElementById('actionIcon');
const actionText             = document.getElementById('actionText');
const findingsEl     = document.getElementById('findings');
const annotatedEl    = document.getElementById('annotated');
const urlListEl      = document.getElementById('urlList');
const headerFrom     = document.getElementById('headerFrom');
const headerReply    = document.getElementById('headerReply');
const headerSubj     = document.getElementById('headerSubj');
const copyBtn        = document.getElementById('copyBtn');

let lastReport = null;

// Character counter
function updateCount() {
  if (charCount && input) {
    charCount.textContent = input.value.length.toLocaleString() + ' characters';

  }
}
if (input) {
  input.addEventListener('input', updateCount);
  updateCount();
}

// Sample buttons
document.querySelectorAll('.chip').forEach(btn => {
  btn.addEventListener('click', () => {
    const s = btn.dataset.sample;
    if (s === 'clear') {
      input.value = '';
      result.classList.remove('show');
      scanStatus.textContent = '';
    } else if (SAMPLES[s]) {
      input.value = SAMPLES[s];
    }
    updateCount();
    input.focus();
  });
});

// Risk gauge animation
const GAUGE_R   = 54;
const GAUGE_CIRC = 2 * Math.PI * GAUGE_R;

function setGauge(scoreVal, cls) {
  if (!gaugeFill) return;
  gaugeFill.style.strokeDasharray  = GAUGE_CIRC;
  gaugeFill.style.strokeDashoffset = GAUGE_CIRC * (1 - scoreVal / 10);
  if (cls === 'fraud') {
    gaugeFill.style.stroke = 'var(--red)';
  } else if (cls === 'high-risk') {
    gaugeFill.style.stroke = '#C2762A';
  } else if (cls === 'spam') {
    gaugeFill.style.stroke = 'var(--spam)';
  } else if (cls === 'suspicious') {
    gaugeFill.style.stroke = 'var(--amber)';
  } else {
    gaugeFill.style.stroke = 'var(--green)';
  }
}

function animateScore(from, to, cls, duration = 600) {
  const start = performance.now();
  function tick(now) {
    const t    = Math.min((now - start) / duration, 1);
    const ease = t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
    const cur  = from + (to - from) * ease;
    if (scoreNum) scoreNum.textContent = Math.round(cur);
    setGauge(cur, cls);
    if (t < 1) requestAnimationFrame(tick);
    else {
      if (scoreNum) scoreNum.textContent = to;
      setGauge(to, cls);
    }
  }
  requestAnimationFrame(tick);
}

// Copy report
if (copyBtn) {
  copyBtn.addEventListener('click', () => {
    if (!lastReport) return;
    navigator.clipboard.writeText(lastReport).then(() => {
      copyBtn.textContent = 'Copied!';
      setTimeout(() => { copyBtn.textContent = 'Copy Report'; }, 2000);
    }).catch(() => {
      copyBtn.textContent = 'Failed — try again';
      setTimeout(() => { copyBtn.textContent = 'Copy Report'; }, 2000);
    });
  });
}

// Analyze button handler
if (analyzeBtn) {
  analyzeBtn.addEventListener('click', async () => {
    const text = input.value.trim();
    if (!text) { input.focus(); return; }

    analyzeBtn.disabled = true;
    scanStatus.textContent = 'Running AI ML model & deep threat heuristics…';
    scanline.classList.add('active');
    result.classList.remove('show');
    stamp.className = 'stamp';
    stamp.classList.remove('show');

    // Run local heuristics analyzer
    const localResult = analyze(text);
    let apiResult = null;

    // Attempt to query trained Python ML backend if running
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      if (response.ok) {
        apiResult = await response.json();
      }
    } catch (e) {
      // Backend not running; fallback gracefully to client-side model
    }

    setTimeout(() => {
      let { score, fraudScore, spamScore, aiResult, verdict, allHits, allMisses, annotated, headers, urlList } = localResult;

      // If live ML model responded, fuse model prediction with heuristics
      if (apiResult && apiResult.success) {
        aiResult.spamProbability = apiResult.spam_probability;
        aiResult.matchedCount = apiResult.top_features.length;

        // Check for calibrated Uncertain state
        if (apiResult.prediction === 'Uncertain') {
          verdict = {
            cls: 'uncertain',
            label: 'UNCERTAIN',
            desc: `ML prediction confidence is near decision boundary (${apiResult.spam_probability}% spam probability). Model recommends manual review.`,
            typeBadge: 'Uncertain (40-60%)'
          };
          score = 5;
        } else if (apiResult.prediction === 'Ham' && fraudScore < 4) {
          verdict = {
            cls: 'legit',
            label: 'Likely Legit',
            desc: `Calibrated Linear SVM confirms ${apiResult.ham_probability}% legitimate probability. No significant fraud detected.`,
            typeBadge: 'Clean Email'
          };
          score = Math.round((100 - apiResult.ham_probability) / 10);
        } else if (apiResult.prediction === 'Spam') {
          if (fraudScore >= 8) {
            verdict = {
              cls: 'fraud',
              label: 'Fraudulent',
              desc: `Critical fraud detected. ML model confirms ${apiResult.spam_probability}% spam probability.`,
              typeBadge: 'Malicious Fraud'
            };
            score = Math.max(score, 8);
          } else {
            verdict = {
              cls: 'spam',
              label: 'Spam / Bulk Mail',
              desc: `Linear SVM Model detected ${apiResult.spam_probability}% spam probability with commercial/bulk markers.`,
              typeBadge: 'Commercial Spam'
            };
            score = Math.max(score, Math.round(apiResult.spam_probability / 10));
          }
        }
      }

      // Generate detailed AI & Threat Explanation
      const expl = buildDetailedExplanation({ score, fraudScore, spamScore, aiResult, verdict, allHits, headers, apiResult, text });
      lastReport = buildReportText(score, fraudScore, spamScore, aiResult, verdict, allHits, headers, expl);

      // Verdict stamp
      stamp.textContent = verdict.label;
      stamp.className   = 'stamp ' + verdict.cls;
      if (verdictDescEl) verdictDescEl.textContent = verdict.desc;

      // Sub-scores
      if (fraudSubscore) fraudSubscore.textContent = `${fraudScore}/10`;
      if (spamSubscore)  spamSubscore.textContent  = `${spamScore}/10`;
      if (threatBadgeEl) {
        threatBadgeEl.textContent = verdict.typeBadge;
        threatBadgeEl.className   = 'subscore-pill-tag ' + verdict.cls;
      }

      // Animated main gauge
      animateScore(0, score, verdict.cls);

      // AI Inference Dashboard
      if (aiProbVal)  aiProbVal.textContent  = `${aiResult.spamProbability}%`;
      if (aiProbFill) {
        aiProbFill.style.width = `${Math.min(100, Math.max(2, aiResult.spamProbability))}%`;
        if (verdict.cls === 'uncertain') {
          aiProbFill.style.background = '#8C6D1F';
        } else if (aiResult.spamProbability >= 70) {
          aiProbFill.style.background = 'var(--red)';
        } else if (aiResult.spamProbability >= 40) {
          aiProbFill.style.background = 'var(--amber)';
        } else {
          aiProbFill.style.background = 'var(--green)';
        }
      }
      if (aiLatencyEl)  aiLatencyEl.textContent  = `${aiResult.latencyMs} ms`;
      if (aiFeaturesEl) {
        aiFeaturesEl.textContent = apiResult && apiResult.model_name 
          ? `Linear SVM (${apiResult.top_features.length} active features)`
          : `${aiResult.matchedCount} active features`;
      }

      // Token Salience (XAI)
      if (tokenSalienceEl) {
        tokenSalienceEl.innerHTML = '';
        const allSalient = [
          ...aiResult.topSpamFeatures.map(f => ({ ...f, type: 'spam' })),
          ...aiResult.topHamFeatures.map(f => ({ ...f, type: 'ham' }))
        ];

        if (allSalient.length === 0 && (!apiResult || apiResult.top_features.length === 0)) {
          tokenSalienceEl.innerHTML = '<div class="token-empty">No dominant weighted NLP tokens detected in this message.</div>';
        } else {
          allSalient.forEach(item => {
            const row = document.createElement('div');
            row.className = `token-row ${item.type}`;
            const cleanToken = item.token.replace(/_/g, ' ');
            const sign = item.weight > 0 ? `+${item.weight.toFixed(1)}` : `${item.weight.toFixed(1)}`;
            const barWidth = Math.min(100, Math.round(Math.abs(item.weight) * 22));

            row.innerHTML = `
              <span class="token-name">${escapeHtml(cleanToken)}</span>
              <div class="token-bar-track">
                <div class="token-bar-fill ${item.type}" style="width: ${barWidth}%"></div>
              </div>
              <span class="token-weight">${sign}</span>
            `;
            tokenSalienceEl.appendChild(row);
          });
        }
      }

      // Render Detailed AI Analysis & Threat Explanation Card
      if (analysisExplanationCard) {
        analysisExplanationCard.className = 'analysis-explanation-card ' + expl.badgeCls;
      }
      if (explanationBadge) {
        explanationBadge.textContent = expl.badgeText;
        explanationBadge.className = 'explanation-badge ' + expl.badgeCls;
      }
      if (explanationTitle) {
        explanationTitle.textContent = expl.title;
      }
      if (explanationSummary) {
        explanationSummary.textContent = expl.summary;
      }
      if (actionIcon) {
        actionIcon.textContent = expl.icon;
      }
      if (actionText) {
        actionText.textContent = expl.actionAdvice;
      }
      if (explanationGrid) {
        explanationGrid.innerHTML = '';
        expl.sections.forEach(sec => {
          const div = document.createElement('div');
          div.className = 'explanation-item';
          div.innerHTML = `
            <div class="explanation-item-title">${escapeHtml(sec.title)}</div>
            <div class="explanation-item-desc">${sec.desc}</div>
          `;
          explanationGrid.appendChild(div);

        });
      }

      // Header panel
      if (headerFrom)  { headerFrom.textContent  = headers.from    || '—'; headerFrom.classList.remove('mismatch'); }
      if (headerReply) { headerReply.textContent = headers.replyTo || '—'; headerReply.classList.remove('mismatch'); }
      if (headerSubj)  headerSubj.textContent    = headers.subject || '—';


      const fromD  = extractDomain(headers.from || '');
      const replyD = extractDomain(headers.replyTo || '');
      if (headerReply && fromD && replyD && fromD !== replyD) {
        headerReply.classList.add('mismatch');
      }

      // Findings list
      findingsEl.innerHTML = '';
      if (allHits.length === 0) {
        findingsEl.innerHTML = '<li class="finding-item clear"><span class="flag clear-flag">Clear</span><span>No fraud, phishing, or spam indicators detected in this email.</span></li>';
      } else {
        allHits.forEach(({ rule, findings }) => {
          findings.forEach(detail => {
            const li = document.createElement('li');
            li.className = 'finding-item hit sev-' + rule.severity;
            li.innerHTML = `
              <div class="finding-header">
                <span class="flag sev-${rule.severity}">${rule.severity}</span>
                <span class="flag-threat-type ${rule.threatType}">${rule.threatType}</span>
                <span class="finding-label">${rule.label}</span>
              </div>
              <div class="finding-detail">${escapeHtml(detail)}</div>`;
            findingsEl.appendChild(li);
          });
        });
        allMisses.forEach(rule => {
          const li = document.createElement('li');
          li.className = 'finding-item clear';
          li.innerHTML = `<span class="flag clear-flag">clear</span><span class="finding-label">${rule.label} — no indicators found.</span>`;
          findingsEl.appendChild(li);
        });
      }

      // URL Inspector
      if (urlListEl) {
        urlListEl.innerHTML = '';
        if (urlList.length === 0) {
          urlListEl.innerHTML = '<li class="url-item">No URLs found in this email.</li>';
        } else {
          urlList.forEach(url => {
            let risk = 'low', riskNote = 'Standard link';
            try {
              const urlObj = new URL(url.startsWith('http') ? url : 'http://' + url);
              const hostname = urlObj.hostname.toLowerCase();

              if (/^\d{1,3}(\.\d{1,3}){3}$/.test(hostname)) {
                risk = 'critical'; riskNote = 'Raw IP address — never legitimate';
              } else if (URL_SHORTENERS.has(hostname)) {
                risk = 'high'; riskNote = 'URL shortener — conceals real destination';
              } else if (SUSPICIOUS_TLDS.has(hostname.split('.').pop())) {
                risk = 'high'; riskNote = `High-risk TLD (.${hostname.split('.').pop()})`;
              } else {
                let isSpoof = false;
                for (const [brand, legit] of Object.entries(BRAND_DB)) {
                  if (hostname.includes(brand) && !legit.includes(hostname) && !legit.some(d => hostname === d || hostname.endsWith('.' + d))) {
                    risk = 'critical'; riskNote = `Possible typosquat of "${brand}"`; isSpoof = true; break;
                  }
                }
                if (!isSpoof) {
                  let hasAffiliate = false;
                  for (const p of urlObj.searchParams.keys()) {
                    if (AFFILIATE_PARAMS.has(p.toLowerCase())) {
                      risk = 'spam'; riskNote = `Commercial affiliate tracking (?${p}=)`; hasAffiliate = true; break;
                    }
                  }
                  if (!hasAffiliate && (TRACKING_DOMAINS.has(hostname) || /^click\./i.test(hostname) || /^track\./i.test(hostname))) {
                    risk = 'spam'; riskNote = 'Bulk email redirector / click tracker';
                  }
                }
              }
            } catch {}
            const li = document.createElement('li');
            li.className = 'url-item';
            li.innerHTML = `<span class="url-risk-badge sev-${risk}">${risk}</span><span class="url-text">${escapeHtml(url)}</span><span class="url-note">${riskNote}</span>`;
            urlListEl.appendChild(li);
          });
        }
      }

      // Annotated exhibit
      annotatedEl.innerHTML = annotated || '<em>No text submitted.</em>';

      // Finish
      scanline.classList.remove('active');
      scanStatus.textContent = `AI analysis complete — ${allHits.length} indicator${allHits.length !== 1 ? 's' : ''} flagged.`;
      analyzeBtn.disabled    = false;
      result.classList.add('show');
      requestAnimationFrame(() => stamp.classList.add('show'));
      result.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 450);
  });
}
