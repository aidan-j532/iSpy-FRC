import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

const nextSteps = [
  {
    number: '01',
    title: 'Get iSpy running',
    text: 'Install the Python package, create the first config and model folders, then start the vision loop.',
    link: '/docs/installation',
    label: 'Install iSpy',
  },
  {
    number: '02',
    title: 'Bring up a camera',
    text: 'Choose a real camera source, check the live image, and start with a pipeline that fits the job.',
    link: '/docs/getting-started',
    label: 'Set up a camera',
  },
  {
    number: '03',
    title: 'Make the numbers useful',
    text: 'Set the camera mount, calibrate what the selected pipeline needs, and check the output against a known target.',
    link: '/docs/calibration',
    label: 'Calibrate',
  },
];

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <p className={styles.eyebrow}>Computer vision for FIRST Robotics Competition</p>
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className={styles.heroCopy}>
          A camera-to-robot vision system for teams that want to see what the
          robot sees, tune it at the field, and decide what to do with the
          results.
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/installation">
            Get started
          </Link>
          <Link
            className={clsx('button button--outline button--secondary button--lg', styles.secondaryButton)}
            to="/docs/architecture">
            How it works
          </Link>
        </div>
      </div>
    </header>
  );
}

function StepCard({number, title, text, link, label}) {
  return (
    <article className={clsx('col col--4', styles.column)}>
      <div className={clsx('card', styles.card)}>
        <div className="card__body">
          <span className={styles.stepNumber}>{number}</span>
          <Heading as="h3">{title}</Heading>
          <p>{text}</p>
        </div>
        <div className={clsx('card__footer', styles.cardFooter)}>
          <Link to={link}>{label} <span aria-hidden="true">→</span></Link>
        </div>
      </div>
    </article>
  );
}

function FlowStep({number, title, text}) {
  return (
    <div className={styles.flowStep}>
      <span className={styles.flowNumber}>{number}</span>
      <div>
        <Heading as="h3">{title}</Heading>
        <p>{text}</p>
      </div>
    </div>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="A camera-to-robot vision system for FRC: configure cameras, run vision pipelines, calibrate measurements, and publish results to NetworkTables.">
      <HomepageHeader />
      <main>
        <section className={clsx('padding-vert--xl', styles.introSection)}>
          <div className="container">
            <div className="row">
              <div className={clsx('col col--7', styles.introCopy)}>
                <p className={styles.eyebrow}>What iSpy is for</p>
                <Heading as="h2">Vision you can work on at the robot</Heading>
                <p>
                  iSpy runs one or more camera pipelines on a coprocessor or
                  development machine. Depending on the pipeline, it can detect
                  objects, read AprilTags or QR codes, estimate depth, or track
                  image motion. For a model-backed detector, it can turn a
                  detection into a measured position using the camera setup and
                  calibration data.
                </p>
                <p>
                  A local web app gives you a place to configure cameras and
                  models, inspect the live feed, watch health and timing, and
                  review recorded runs. When you enable the NetworkTables
                  utility, the same running loop can publish selected results
                  to robot code.
                </p>
                <p>
                  It is not a magic camera or a substitute for testing on the
                  field. The camera, lens, mount, model, lighting, and target
                  all affect the result. iSpy puts those pieces in one system
                  and gives you tools to see where things are going wrong.
                </p>
                <Link to="/docs/architecture">Read the runtime overview →</Link>
              </div>
              <aside className={clsx('col col--5', styles.calloutColumn)}>
                <div className={styles.callout}>
                  <Heading as="h3">A useful mental model</Heading>
                  <p>
                    iSpy has three jobs: get a frame, turn it into a consistent
                    result, then make that result useful to a person or the
                    robot. A camera can be healthy while its detections are
                    wrong, and detections can look right while their measured
                    positions are off. Check each step on its own.
                  </p>
                </div>
              </aside>
            </div>
          </div>
        </section>

        <section className={styles.flowSection}>
          <div className="container">
            <div className={styles.sectionHeading}>
              <p className={styles.eyebrow}>From pixels to robot code</p>
              <Heading as="h2">How the pieces fit together</Heading>
              <p>
                A camera is assigned a pipeline in the config. The runtime
                processes frames continuously, and the web app and enabled
                utilities receive the results from that loop.
              </p>
            </div>
            <div className={styles.flowGrid}>
              <FlowStep
                number="1"
                title="Capture"
                text="A camera source supplies frames. iSpy can run a single camera or combine results from several."
              />
              <FlowStep
                number="2"
                title="Process"
                text="The selected pipeline detects or measures what matters. Model-backed pipelines use the model and backend configured for that camera."
              />
              <FlowStep
                number="3"
                title="Interpret"
                text="Camera mounting values and the calibration needed by the pipeline turn image measurements into more useful coordinates."
              />
              <FlowStep
                number="4"
                title="Inspect and publish"
                text="The dashboard shows feeds, status, and metrics. Configured utilities can pass results to NetworkTables or other consumers."
              />
            </div>
            <p className={styles.flowNote}>
              Calibration is not a model setting, and it cannot fix a poor
              detection. Get a stable live image first, then confirm detections,
              then check the geometry.
            </p>
          </div>
        </section>

        <section className={clsx('padding-vert--xl', styles.stepsSection)}>
          <div className="container">
            <div className={styles.sectionHeading}>
              <p className={styles.eyebrow}>A sensible first session</p>
              <Heading as="h2">Start small, then tune</Heading>
              <p>
                One camera and one pipeline are enough to learn the system.
                Once that is solid, add calibration, robot output, or another
                camera.
              </p>
            </div>
            <div className="row">
              {nextSteps.map((step) => (
                <StepCard key={step.number} {...step} />
              ))}
            </div>
          </div>
        </section>

        <section className={styles.audienceSection}>
          <div className="container">
            <div className="row">
              <div className={clsx('col col--6', styles.audienceBlock)}>
                <Heading as="h2">For teams setting up a robot</Heading>
                <p>
                  The recommended deployment target is an Orange Pi 5 or 5 Pro
                  with the RK3588 NPU. Other Linux boards and development
                  computers can work too, but camera support and inference
                  speed depend on the operating system, installed runtime, and
                  model format. Start with the <Link to="/docs/installation">installation guide</Link>
                  and check the backend notes before choosing hardware.
                </p>
                <Link to="/docs/vision-pipelines">Compare pipelines and backends →</Link>
              </div>
              <div className={clsx('col col--6', styles.audienceBlock)}>
                <Heading as="h2">For people building iSpy</Heading>
                <p>
                  The Python package and this documentation site are separate
                  projects in the repository. The runtime is organized around
                  per-camera pipelines, optional plugins, and a local Flask
                  dashboard. The developer guide covers editable installs,
                  extension points, and how to work on the docs site.
                </p>
                <Link to="/docs/development-and-advanced-config">Developer guide →</Link>
              </div>
            </div>
          </div>
        </section>

        <section className={clsx('padding-vert--xl', styles.moreSection)}>
          <div className="container">
            <div className={styles.sectionHeading}>
              <p className={styles.eyebrow}>Keep going</p>
              <Heading as="h2">Useful when you need it</Heading>
            </div>
            <div className={styles.linkGrid}>
              <Link to="/docs/configuration">Understand the config</Link>
              <Link to="/docs/cameras-and-models">Choose cameras and models</Link>
              <Link to="/docs/datasets-and-recordings">Work with datasets and recordings</Link>
              <Link to="/docs/web-ui-and-networking">Connect the dashboard and robot</Link>
              <Link to="/docs/troubleshooting">Troubleshoot a live setup</Link>
              <Link to="/docs/cli-reference">Look up commands</Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
