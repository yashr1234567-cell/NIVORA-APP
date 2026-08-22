import { useRouter } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';

import { ExternalLink } from '@/components/external-link';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Collapsible } from '@/components/ui/collapsible';
import { BottomTabInset, MaxContentWidth, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

export default function ExploreScreen() {
  const router = useRouter();
  const theme = useTheme();
  const insets = useSafeAreaInsets();

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.background }]}>
      <ScrollView
        contentContainerStyle={[
          styles.scrollContainer,
          { paddingBottom: insets.bottom + BottomTabInset + Spacing.four },
        ]}
        showsVerticalScrollIndicator={false}
      >
        {/* Header Section */}
        <View style={styles.header}>
          <View style={styles.badgeRow}>
            <View style={styles.badge}>
              <Text style={styles.badgeText}>RESEARCH & CLINICAL EVIDENCE</Text>
            </View>
          </View>
          <ThemedText style={styles.mainTitle}>Acoustic Biomarkers & UCI Dataset</ThemedText>
          <ThemedText style={styles.subTitle} themeColor="textSecondary">
            Comprehensive guide to voice analysis, digital biomarkers, and machine learning models for early detection of Parkinson&apos;s disease.
          </ThemedText>
        </View>

        {/* Highlight Stats Banner */}
        <ThemedView type="backgroundElement" style={styles.statsBanner}>
          <View style={styles.statBox}>
            <Text style={styles.statNum}>756</Text>
            <Text style={styles.statLabel}>Patient Samples</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statBox}>
            <Text style={styles.statNum}>755</Text>
            <Text style={styles.statLabel}>Acoustic Features</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statBox}>
            <Text style={styles.statNum}>0.909</Text>
            <Text style={styles.statLabel}>Diagnostic AUC</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statBox}>
            <Text style={styles.statNum}>10s</Text>
            <Text style={styles.statLabel}>Sustained /a/ Test</Text>
          </View>
        </ThemedView>

        {/* Collapsible Research Sections */}
        <View style={styles.sectionsContainer}>
          {/* 1. UCI Dataset Overview */}
          <Collapsible title="1. UCI Machine Learning Dataset Specification">
            <ThemedView type="backgroundElement" style={styles.collapsibleContent}>
              <ThemedText style={styles.bodyText}>
                The speech dataset was gathered from 188 patients with Parkinson&apos;s Disease (107 men and 81 women) and 64 healthy controls (23 men and 41 women). Each subject uttered the sustained vowel sound &quot;a&quot; three times, resulting in 756 total recordings.
              </ThemedText>

              <View style={styles.dataGrid}>
                <View style={styles.dataRow}>
                  <Text style={styles.dataKey}>Parkinson&apos;s Disease Cohort</Text>
                  <Text style={styles.dataVal}>564 recordings (74.6%)</Text>
                </View>
                <View style={styles.dataRow}>
                  <Text style={styles.dataKey}>Healthy Control Cohort</Text>
                  <Text style={styles.dataVal}>192 recordings (25.4%)</Text>
                </View>
                <View style={styles.dataRow}>
                  <Text style={styles.dataKey}>Audio Format</Text>
                  <Text style={styles.dataVal}>44.1 kHz / 16-bit Mono</Text>
                </View>
                <View style={styles.dataRow}>
                  <Text style={styles.dataKey}>Primary Reference</Text>
                  <Text style={styles.dataVal}>Sakar et al., IEEE JBHI (2019)</Text>
                </View>
              </View>

              <ExternalLink href="https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification">
                <ThemedText type="linkPrimary">View UCI Dataset on UCI ML Repository →</ThemedText>
              </ExternalLink>
            </ThemedView>
          </Collapsible>

          {/* 2. Clinical Biomarkers */}
          <Collapsible title="2. Acoustic Biomarkers & Clinical Thresholds">
            <ThemedView type="backgroundElement" style={styles.collapsibleContent}>
              <ThemedText style={styles.bodyText}>
                Hypokinetic dysarthria manifests as micro-perturbations in pitch, amplitude stability, and vocal fold adduction:
              </ThemedText>

              <View style={styles.biomarkerCardsList}>
                {/* Jitter */}
                <View style={styles.biomarkerItem}>
                  <View style={styles.biomarkerHeader}>
                    <Text style={styles.biomarkerName}>Local Jitter (%)</Text>
                    <Text style={styles.normTag}>Norm: &lt; 1.04%</Text>
                  </View>
                  <ThemedText type="small" themeColor="textSecondary">
                    Measures cycle-to-cycle frequency variation. Laryngeal muscle rigidity leads to subglottic pressure fluctuations and elevated pitch jitter.
                  </ThemedText>
                </View>

                {/* Shimmer */}
                <View style={styles.biomarkerItem}>
                  <View style={styles.biomarkerHeader}>
                    <Text style={styles.biomarkerName}>Local Shimmer (%)</Text>
                    <Text style={styles.normTag}>Norm: &lt; 3.81%</Text>
                  </View>
                  <ThemedText type="small" themeColor="textSecondary">
                    Quantifies cycle-to-cycle amplitude perturbation. Incomplete vocal cord closure results in volume variability and vocal fatigue.
                  </ThemedText>
                </View>

                {/* HNR */}
                <View style={styles.biomarkerItem}>
                  <View style={styles.biomarkerHeader}>
                    <Text style={styles.biomarkerName}>Harmonics-to-Noise Ratio (HNR)</Text>
                    <Text style={styles.normTag}>Norm: &gt; 20 dB</Text>
                  </View>
                  <ThemedText type="small" themeColor="textSecondary">
                    Ratio of harmonic acoustic energy to turbulent breath noise. Parkinsonian voice drops below 20 dB due to breathiness and hoarseness.
                  </ThemedText>
                </View>

                {/* PPE */}
                <View style={styles.biomarkerItem}>
                  <View style={styles.biomarkerHeader}>
                    <Text style={styles.biomarkerName}>Pitch Period Entropy (PPE)</Text>
                    <Text style={styles.normTag}>Norm: &lt; 0.20</Text>
                  </View>
                  <ThemedText type="small" themeColor="textSecondary">
                    Non-linear entropy metric reflecting natural frequency variations. Higher values indicate difficulty sustaining a constant pitch.
                  </ThemedText>
                </View>
              </View>
            </ThemedView>
          </Collapsible>

          {/* 3. Feature Extraction Pipeline */}
          <Collapsible title="3. 755-Feature Acoustic Extraction Pipeline">
            <ThemedView type="backgroundElement" style={styles.collapsibleContent}>
              <ThemedText style={styles.bodyText}>
                The feature extraction pipeline captures multi-resolution signal dynamics across time, frequency, and wavelet domains:
              </ThemedText>

              <View style={styles.pipelineSteps}>
                <View style={styles.pipelineStep}>
                  <Text style={styles.stepBadge}>A</Text>
                  <View style={styles.stepContent}>
                    <ThemedText type="smallBold">Baseline Time-Domain Perturbations (21 Features)</ThemedText>
                    <ThemedText type="small" themeColor="textSecondary">
                      Jitter variants (locAbs, rap, ppq5, ddp), Shimmer variants (locDb, apq3, apq5, apq11, dda), and fundamental frequency moments.
                    </ThemedText>
                  </View>
                </View>

                <View style={styles.pipelineStep}>
                  <Text style={styles.stepBadge}>B</Text>
                  <View style={styles.stepContent}>
                    <ThemedText type="smallBold">Formants & Glottal Flow (16 Features)</ThemedText>
                    <ThemedText type="small" themeColor="textSecondary">
                      Formant resonant frequencies (F1–F4), bandwidths (B1–B4), Glottal Quasiclosed (GQ) periods, and Vocal Fold Excitation (VFER).
                    </ThemedText>
                  </View>
                </View>

                <View style={styles.pipelineStep}>
                  <Text style={styles.stepBadge}>C</Text>
                  <View style={styles.stepContent}>
                    <ThemedText type="smallBold">Mel-Frequency Cepstral Coefficients (84 Features)</ThemedText>
                    <ThemedText type="small" themeColor="textSecondary">
                      Mean and standard deviation of 13 MFCC coefficients, first derivatives (delta), and second derivatives (delta-delta).
                    </ThemedText>
                  </View>
                </View>

                <View style={styles.pipelineStep}>
                  <Text style={styles.stepBadge}>D</Text>
                  <View style={styles.stepContent}>
                    <ThemedText type="smallBold">Wavelet & TQWT Decompositions (634 Features)</ThemedText>
                    <ThemedText type="small" themeColor="textSecondary">
                      Tunable Q-Factor Wavelet Transform across 36 sub-bands extracting sub-band energies, Shannon entropy, log entropy, and TKEO dynamics.
                    </ThemedText>
                  </View>
                </View>
              </View>
            </ThemedView>
          </Collapsible>

          {/* 4. Machine Learning & ROC AUC */}
          <Collapsible title="4. Machine Learning Classification & Diagnostics">
            <ThemedView type="backgroundElement" style={styles.collapsibleContent}>
              <ThemedText style={styles.bodyText}>
                Supervised machine learning algorithms trained on the dataset demonstrate high discriminative power for non-invasive remote screening:
              </ThemedText>

              <View style={styles.mlTable}>
                <View style={styles.mlRowHeader}>
                  <Text style={[styles.mlCell, styles.mlCellHeader, { flex: 1.8 }]}>Classifier</Text>
                  <Text style={[styles.mlCell, styles.mlCellHeader]}>Accuracy</Text>
                  <Text style={[styles.mlCell, styles.mlCellHeader]}>Sensitivity</Text>
                  <Text style={[styles.mlCell, styles.mlCellHeader]}>ROC AUC</Text>
                </View>
                <View style={styles.mlRow}>
                  <Text style={[styles.mlCell, { flex: 1.8, fontWeight: '700' }]}>Random Forest (100 Trees)</Text>
                  <Text style={styles.mlCell}>87.3%</Text>
                  <Text style={styles.mlCell}>91.2%</Text>
                  <Text style={[styles.mlCell, styles.highlightScore]}>0.909</Text>
                </View>
                <View style={styles.mlRow}>
                  <Text style={[styles.mlCell, { flex: 1.8 }]}>Support Vector Machine (RBF)</Text>
                  <Text style={styles.mlCell}>85.8%</Text>
                  <Text style={styles.mlCell}>88.7%</Text>
                  <Text style={styles.mlCell}>0.892</Text>
                </View>
                <View style={styles.mlRow}>
                  <Text style={[styles.mlCell, { flex: 1.8 }]}>XGBoost Gradient Boosted</Text>
                  <Text style={styles.mlCell}>86.5%</Text>
                  <Text style={styles.mlCell}>90.4%</Text>
                  <Text style={styles.mlCell}>0.901</Text>
                </View>
              </View>

              <ThemedText type="small" themeColor="textSecondary" style={styles.noteText}>
                Evaluation performed with stratified 5-fold cross-validation and subject-independent train/test partitioning.
              </ThemedText>
            </ThemedView>
          </Collapsible>

          {/* 5. Clinical Testing Protocol */}
          <Collapsible title="5. Standardized Phonation Screening Protocol">
            <ThemedView type="backgroundElement" style={styles.collapsibleContent}>
              <ThemedText style={styles.bodyText}>
                To ensure maximum signal-to-noise ratio and reproducible acoustic extraction:
              </ThemedText>

              <View style={styles.protocolList}>
                <View style={styles.protocolItem}>
                  <Text style={styles.protocolBullet}>•</Text>
                  <ThemedText style={styles.protocolText}>
                    <Text style={styles.boldText}>Environment:</Text> Ambient noise must be below 40 dBA. Ensure no background conversation or echoes.
                  </ThemedText>
                </View>
                <View style={styles.protocolItem}>
                  <Text style={styles.protocolBullet}>•</Text>
                  <ThemedText style={styles.protocolText}>
                    <Text style={styles.boldText}>Microphone Positioning:</Text> Maintain a 15–20 cm (6–8 inch) distance at a 45° angle to prevent breath clipping.
                  </ThemedText>
                </View>
                <View style={styles.protocolItem}>
                  <Text style={styles.protocolBullet}>•</Text>
                  <ThemedText style={styles.protocolText}>
                    <Text style={styles.boldText}>Phonation Duration:</Text> Inhale deeply and sustain a steady &quot;AAAAH&quot; vowel for at least 8 to 10 seconds at a comfortable, natural pitch.
                  </ThemedText>
                </View>
                <View style={styles.protocolItem}>
                  <Text style={styles.protocolBullet}>•</Text>
                  <ThemedText style={styles.protocolText}>
                    <Text style={styles.boldText}>Repetition:</Text> Collect three consecutive trials and average the acoustic perturbation parameters for longitudinal tracking.
                  </ThemedText>
                </View>
              </View>
            </ThemedView>
          </Collapsible>
        </View>

        {/* CTA Launch Screening */}
        <ThemedView type="backgroundElement" style={styles.ctaCard}>
          <ThemedText style={styles.ctaTitle}>Ready to Test Vocal Biomarkers?</ThemedText>
          <ThemedText type="small" themeColor="textSecondary" style={styles.ctaSub}>
            Record a 10-second sustained vowel sample or test against clinical research presets.
          </ThemedText>

          <Pressable
            style={({ pressed }) => [styles.ctaButton, pressed && styles.pressed]}
            onPress={() => router.push('/parkinsons')}
          >
            <Text style={styles.ctaButtonText}>Open Voice Screening Module 🎙️</Text>
          </Pressable>
        </ThemedView>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
  },
  scrollContainer: {
    paddingHorizontal: Spacing.four,
    paddingVertical: Spacing.four,
    maxWidth: MaxContentWidth,
    alignSelf: 'center',
    width: '100%',
    gap: Spacing.four,
  },
  header: {
    gap: Spacing.two,
  },
  badgeRow: {
    flexDirection: 'row',
  },
  badge: {
    backgroundColor: '#0284C7',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  badgeText: {
    color: '#FFFFFF',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  mainTitle: {
    fontSize: 24,
    fontWeight: '700',
    letterSpacing: -0.4,
  },
  subTitle: {
    fontSize: 14,
    lineHeight: 21,
  },
  statsBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-around',
    padding: Spacing.three,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#BAE6FD',
    backgroundColor: '#F0F9FF',
  },
  statBox: {
    alignItems: 'center',
  },
  statNum: {
    fontSize: 20,
    fontWeight: '800',
    color: '#0284C7',
  },
  statLabel: {
    fontSize: 11,
    color: '#475569',
    marginTop: 2,
    fontWeight: '500',
  },
  statDivider: {
    width: 1,
    height: 32,
    backgroundColor: '#BAE6FD',
  },
  sectionsContainer: {
    gap: Spacing.three,
  },
  collapsibleContent: {
    padding: Spacing.three,
    borderRadius: 12,
    gap: Spacing.three,
    marginTop: Spacing.two,
  },
  bodyText: {
    fontSize: 14,
    lineHeight: 21,
    color: '#334155',
  },
  dataGrid: {
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#E2E8F0',
    overflow: 'hidden',
  },
  dataRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#F1F5F9',
  },
  dataKey: {
    fontSize: 13,
    color: '#475569',
    fontWeight: '500',
  },
  dataVal: {
    fontSize: 13,
    fontWeight: '700',
    color: '#0F172A',
  },
  biomarkerCardsList: {
    gap: 8,
  },
  biomarkerItem: {
    backgroundColor: '#FFFFFF',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#E2E8F0',
    gap: 4,
  },
  biomarkerHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  biomarkerName: {
    fontSize: 13,
    fontWeight: '700',
    color: '#0F172A',
  },
  normTag: {
    fontSize: 11,
    fontWeight: '700',
    color: '#0284C7',
    backgroundColor: '#E0F2FE',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  pipelineSteps: {
    gap: 10,
  },
  pipelineStep: {
    flexDirection: 'row',
    gap: 10,
    alignItems: 'flex-start',
  },
  stepBadge: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: '#0284C7',
    color: '#FFFFFF',
    textAlign: 'center',
    lineHeight: 22,
    fontSize: 11,
    fontWeight: '700',
    marginTop: 2,
  },
  stepContent: {
    flex: 1,
    gap: 2,
  },
  mlTable: {
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#E2E8F0',
    overflow: 'hidden',
  },
  mlRowHeader: {
    flexDirection: 'row',
    backgroundColor: '#F8FAFC',
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#E2E8F0',
  },
  mlRow: {
    flexDirection: 'row',
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#F1F5F9',
    alignItems: 'center',
  },
  mlCell: {
    flex: 1,
    fontSize: 12,
    color: '#334155',
    textAlign: 'center',
  },
  mlCellHeader: {
    fontWeight: '700',
    color: '#0F172A',
  },
  highlightScore: {
    color: '#0284C7',
    fontWeight: '800',
  },
  noteText: {
    fontSize: 12,
    fontStyle: 'italic',
  },
  protocolList: {
    gap: 8,
  },
  protocolItem: {
    flexDirection: 'row',
    gap: 8,
    alignItems: 'flex-start',
  },
  protocolBullet: {
    color: '#0284C7',
    fontWeight: '800',
    fontSize: 14,
  },
  protocolText: {
    fontSize: 13,
    lineHeight: 19,
    flex: 1,
    color: '#334155',
  },
  boldText: {
    fontWeight: '700',
    color: '#0F172A',
  },
  ctaCard: {
    borderRadius: 16,
    padding: Spacing.four,
    borderWidth: 1,
    borderColor: '#BAE6FD',
    backgroundColor: '#F0F9FF',
    alignItems: 'center',
    gap: Spacing.two,
    marginTop: Spacing.two,
  },
  ctaTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#0369A1',
  },
  ctaSub: {
    textAlign: 'center',
    marginBottom: Spacing.two,
  },
  ctaButton: {
    backgroundColor: '#0284C7',
    paddingVertical: 14,
    paddingHorizontal: 24,
    borderRadius: 12,
    shadowColor: '#0284C7',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 6,
    elevation: 3,
  },
  ctaButtonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '700',
  },
  pressed: {
    opacity: 0.8,
    transform: [{ scale: 0.98 }],
  },
});

