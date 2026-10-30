# Contributing to iSpy-FRC

Thank you for your interest in contributing to iSpy-FRC!

iSpy-FRC aims to make computer vision more accessible to FIRST Robotics Competition teams across a range of hardware. Contributions help improve camera support, inference backends, performance, reliability, documentation, and the overall experience for teams.

We welcome contributions of all sizes, from fixing a typo to improving hardware acceleration or designing a new subsystem.

## Ways to contribute

You can help by:

- Fixing bugs and reliability issues.
- Improving camera backends and frame acquisition.
- Improving inference backends, including Hailo and other hardware accelerators.
- Improving model and runtime compatibility.
- Improving performance, portability, and resource usage.
- Improving configuration, diagnostics, and error handling.
- Writing documentation, examples, and installation instructions.
- Reporting reproducible bugs and testing on different hardware.
- Reviewing proposed changes and suggesting improvements.

You do not need to contribute code to help the project.

## Before starting a large change

For substantial changes, open an issue or discussion before implementing the feature. Explain the problem, your proposed solution, affected hardware, and any important trade-offs.

This is especially helpful for new camera backends, inference integrations, plugin interfaces, and architectural changes.

For small, self-contained fixes, you can generally open a pull request directly.

Discussing major changes early helps prevent contributors from spending time on work that conflicts with the project's direction.

## Development guidelines

Please aim for code that is:

- **Focused:** Solves a clear problem without unrelated changes.
- **Maintainable:** Uses straightforward designs that future contributors can understand.
- **Compatible:** Preserves existing behavior unless a change is necessary and discussed.
- **Portable:** Considers supported operating systems, Python environments, cameras, and accelerators.
- **Diagnosable:** Provides useful errors when configuration, dependencies, or hardware initialization fails.
- **Documented:** Explains relevant configuration, requirements, limitations, and setup.

Follow the conventions of the surrounding code. Avoid unnecessary dependencies and avoid silently swallowing errors that prevent a feature from working correctly.

## Testing

Test your changes as thoroughly as reasonably possible.

In your pull request, explain:

- What changed and why.
- How you tested it.
- Relevant operating system, Python, hardware, and runtime versions.
- Whether testing used real hardware or a simulated environment.
- Any known limitations or untested configurations.

Do not claim support for hardware you have not tested. It is fine to explain that you lack access to a particular device and request help with verification.

## Pull requests

A pull request should explain the problem, describe the implementation, identify relevant testing, and mention breaking changes or limitations.

Include documentation updates where appropriate and link related issues.

Maintainers may request revisions, further testing, or a narrower scope. A submission is not guaranteed to be merged, but maintainers should communicate significant decisions respectfully and explain them where practical.

## Contributor ownership and licensing

Contributor ownership matters to this project. The goal is to make it possible for developers to improve iSpy-FRC without requiring them to give up copyright ownership or grant commercial relicensing rights upfront.

### You retain ownership

You retain copyright in original work you contribute. Submitting a contribution does not assign your copyright to the project.

You may reuse your own original code in other projects, subject to third-party rights, applicable licenses, and any other obligations that apply to your work.

### Current project license

iSpy-FRC currently uses the **PolyForm Noncommercial License 1.0.0**. This remains the project's license unless the project separately announces a change.

By contributing, you authorize the project to incorporate, modify, reproduce, and distribute your contribution as part of iSpy-FRC under its current license and applicable noncommercial terms.

This does not require you to grant the project permission to relicense your contribution commercially.

### Future commercial licensing

The project may consider offering a separate commercial license in the future. No contributor is required by these contribution terms to authorize that commercial relicensing in advance.

If the project later wants to include your contribution in a commercial or differently licensed distribution, it must obtain any additional permissions required from the relevant rights holders.

You may agree or decline. If permission is not granted, the project may need to omit or replace the contribution in the affected distribution.

### Attribution

The project aims to preserve relevant copyright and third-party license notices and give contributors appropriate recognition through commit history, release notes, documentation, or contributor lists where practical.

Submitting a contribution does not guarantee payment, employment, or a particular form of recognition.

### Read the contributor terms

Please review `CONTRIBUTOR_LICENSE_AGREEMENT.md` before submitting a contribution. It explains the permissions needed for the current project and explicitly excludes an automatic grant of future commercial relicensing rights.

Do not submit work that you do not own or lack permission to contribute. If your contribution includes third-party code, generated material, or other restricted content, disclose its origin and relevant license.

## Security

Do not publish passwords, private keys, access tokens, or sensitive security details in public issues or pull requests.

Use the repository's private security reporting channel if one is available. If no private channel is established, contact the maintainers privately before disclosing sensitive details publicly.

## Conduct

All contributors are expected to follow `CODE_OF_CONDUCT.md`.

Respectful technical disagreement is welcome. Critique ideas and implementations constructively, and treat other contributors fairly.

## Questions

If you have questions about architecture, testing, licensing, or whether a change fits the project, ask before investing substantial effort.

You should have a reasonable opportunity to understand the contribution terms before agreeing to them. You are welcome to ask licensing questions without being pressured to accept additional permissions.

Thank you for helping improve iSpy-FRC!