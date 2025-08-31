---
title: 'OWASP: Things Android Developers should know'
date: 2025-08-02
lastmod: '2025-08-02'
tags: ['android', 'security']
draft: false
summary: 'As mobile developers, it’s not enough to just build functional apps; we must ensure their security. Previously, app security often seemed like a backend-only concern, handled by measures like authentication, authorization, and TLS.'
---

As mobile developers, it’s not enough to just build functional apps; we must ensure their security. Previously, app security often seemed like a backend-only concern, handled by measures like authentication, authorization, and TLS.

Backend security is crucial, but it’s equally vital to understand mobile app security. After all, the user’s first interaction is always through the app. What if the users are unaware their devices are infected with malware, which can then make our applications vulnerable.

Imagine the user install unknown app from unofficial store on their laptop. They might not realize it contains a keylogger. Then, when user access websites and make transactions, the keylogger secretly records your username, password, and other sensitive information without you ever knowing. It will leads to another vulnerabilities even though we securing our BE side with full power.

Just like computers, smartphones also have operating systems that allow processes to run in the background unnoticed by the user. Therefore, smartphones are susceptible to the same vulnerabilities as computers.

When your application operates on an insecure device, the risk of attacks stemming from user negligence significantly increases. This can expose your app to a range of threats, from SMS logging — where malicious software intercepts one-time passwords (OTPs) sent via text — to keyloggers that record every keystroke, potentially stealing sensitive data. Other vulnerabilities like tapjacking, which tricks users into tapping on unintended elements, also become serious concerns. Essentially, if the underlying device isn’t secure, your app becomes a potential victim of these intrusive methods, compromising user data and application integrity.

What is OWASP? A Foundation for Secure Development

So, what exactly is OWASP? OWASP (Open Worldwide Application Security Project) is a non-profit organization dedicated to establishing security standards and improving software security. You can find more detailed information about OWASP.

Why knowing OWASP Matters?

In software development, we often feel insecure about the applications we’re building. Remember, we don’t know what we don’t know. That’s precisely why we need to understand security standards. Security is a cat-and-mouse game; vulnerabilities will always emerge. Just because we can patch a security hole today doesn’t mean new ones won’t appear tomorrow.

Google, as a principal authority in Android development, also recommends OWASP as a benchmark for securing Android applications.

OWASP provides a dedicated sub-field that specifically delves into the intricacies of mobile application security. This specialized area is widely recognized as OWASP Mobile Application Security (OWASP MAS). Within its comprehensive framework, you’ll find a variety of crucial topics, including:

- Standard for privacy (OWASP MASVS)
- Collection of mobile application Vulnarabillities (OWASP MASWE),
- How to test mobile security (OWASP MASTG).

Here are some of the standardization categories:

- Storage
- Crypto
- Authentication
- Network
- Platform
- Code
- Resilience
- Privacy

Beyond offering standardization, vulnerability insights, and mobile app testing methods, what’s most compelling about OWASP is the playground on their website. It allows us to practice testing applications through reverse engineering. This provides a practical way to learn how to find security flaws in apps. With consistent practice, you can then apply these skills to your own applications.

Here are some of the knowledge and best practices you can find on the OWASP website, including:

### Code Obfuscation

Code obfuscation is the process of modifying an app’s binary to make it harder for humans to understand. Obfuscation hides function and class names in your compiled code, replacing each symbol with another symbol, making it difficult for an attacker to reverse engineer your proprietary app. (Detail)

### Runtime Integrity Verification

Your app might seem secure because some data is stored using NDK (C/C++), making it hard to find through reverse engineering. However, attackers can manipulate data at runtime. That’s why it’s crucial to also check your app’s integrity during runtime. (Detail)

### Secure Shared Preference

SharedPreferences are typically used to store data in an XML file format. Their advantage lies in their key-value structure, which makes them easy to access for simple data like usernames, login status, and so on. However, for sensitive data, you should always use EncryptedSharedPreferences. (Detail)

### Google Play Integrity

Google Play Integrity (formerly SafetyNet) is used to ensure your application is running in a secure device environment. For example, it helps detect rooted devices or apps downloaded from outside Google Play. (Detail)

### Permission Handling

Before adding any permissions to your app, it’s crucial to understand why that permission is truly needed. For instance, if you have a QR feature, you only need camera access. In this case, it’s best to use runtime permissions, meaning the permission is only requested when the feature is actually used.
Moreover, if your app only needs to capture images, consider using a built-in camera app. This way, your app won’t require camera access directly. (Detail)

# Conclusion

The examples above are just a few of the minimum requirements to help secure your application. The OWASP website provides much more detailed information on various types of security vulnerabilities and how to address them.

Finally, always remember: security is a cat-and-mouse game. No matter how robust our defenses, there will always be new vulnerabilities we need to watch out for.

Happy coding ~~
