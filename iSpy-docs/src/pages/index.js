import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Read the docs
          </Link>
        </div>
      </div>
    </header>
  );
}

function FeatureCard({title, text}) {
  return (
    <div className="col col--4">
      <div className="card margin-bottom--lg">
        <div className="card__body">
          <Heading as="h3">{title}</Heading>
          <p>{text}</p>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="FRC computer vision pipeline for detection, calibration, and NetworkTables output.">
      <HomepageHeader />
      <main>
        <section className="padding-top--xl padding-bottom--xl">
          <div className="container">
            <div className="row">
              <FeatureCard
                title="Field-aware vision"
                text="Watch a camera, detect objects, and convert those detections into real-world field coordinates for robot use."
              />
              <FeatureCard
                title="Built for FRC"
                text="Publish data over NetworkTables, keep a local dashboard for tuning, and work with standard robot workflows."
              />
              <FeatureCard
                title="Runs on real hardware"
                text="Support for Orange Pi deployments, ARM64 boards, x86 Linux desktops, and other common deployment targets."
              />
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
