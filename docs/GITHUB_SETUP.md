# راهنمای ساخت صفحه GitHub حرفه‌ای برای Omage

## ۱. ساختار Repository

```
omage/
├── .github/
│   ├── workflows/
│   │   └── ci.yml          ← CI/CD
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
├── docs/
│   └── logo.png            ← لوگو رو اینجا بذار
├── examples/
├── omage/
├── tests/
├── README.md               ← صفحه اصلی GitHub
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
└── pyproject.toml
```

## ۲. تنظیمات Repository

بعد از push کردن:

1. برو Settings → About (گوشه راست بالا)
2. اینا رو پر کن:
   - Description: `Build AI the way you think.`
   - Website: (اگه داری)
   - Topics: `python`, `ai`, `machine-learning`, `deep-learning`, `pytorch`, `neural-network`, `open-source`

## ۳. Release ساختن

```bash
git tag v0.1.1-alpha
git push origin v0.1.1-alpha
```

بعد برو GitHub → Releases → Create new release:
- Tag: v0.1.1-alpha
- Title: Omage v0.1.1 — Alpha Release
- Description: همون CHANGELOG

## ۴. Badges که توی README هست

همه badges خودکار کار میکنن وقتی CI/CD فعال بشه.

## ۵. لوگو

فایل SVG لوگو رو به PNG تبدیل کن (512×512) و بذار توی docs/logo.png
